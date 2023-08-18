
#include "onnx-parser.hpp"
#include "onnx/onnx-ml.pb.h"
#include "onnx/onnx-operators-ml.pb.h"
#include <fstream>
#include <numeric>
#include <unordered_map>

namespace spconv{

struct ParameterFP16Data{
    std::vector<unsigned short> data;
    std::vector<int> shape;
};

static onnx::TensorProto get_initializer(const onnx::GraphProto& graph, const std::string& name) {
    for (int i = 0; i < graph.initializer_size(); ++i) {
        auto& init = graph.initializer(i);
        if (init.name() == name) 
            return init;
    }
    printf("Can not find initializer '%s' in ONNX.\n", name.c_str());
    return onnx::TensorProto();
};

static ParameterFP16Data get_initializer_data(const onnx::GraphProto& graph, const std::string& name) {
    auto proto = get_initializer(graph, name);
    if(proto.data_type() != onnx::TensorProto_DataType_FLOAT16){
        printf("Can not support non float data type[%d] for initializer data.\n", proto.data_type());
        return ParameterFP16Data();
    }

    ParameterFP16Data output;
    output.shape.resize(proto.dims().size());
    std::transform(proto.dims().begin(), proto.dims().end(), output.shape.begin(), [](int64_t x){return (int)x;});

    size_t volumn = std::accumulate(output.shape.begin(), output.shape.end(), 1ul, std::multiplies<int64_t>());
    if(volumn * sizeof(unsigned short) != proto.raw_data().size()){
        printf("Invalid parameter data size.\n");
        return ParameterFP16Data();
    }
    unsigned short* pdata = (unsigned short*)proto.raw_data().data();
    output.data = std::vector<unsigned short>(pdata, pdata + volumn);
    return output;
};

static onnx::AttributeProto get_attribute(const onnx::NodeProto& node, const std::string& name) {
    for (int i = 0; i < node.attribute_size(); ++i) {
        auto& attr = node.attribute(i);
        if (attr.name() == name) 
            return attr;
    }
    printf("Can not find attribute '%s' in node '%s', it will use the default value of.\n",
        name.c_str(), node.name().c_str());
    return onnx::AttributeProto();
};

static std::vector<int> get_attribute_as_intarray(const onnx::NodeProto& node, const std::string& name) {
    auto ints = get_attribute(node, name).ints();
    std::vector<int> output(ints.size());
    for (int i = 0; i < ints.size(); ++i) 
        output[i] = ints[i];
    return output;
};

std::shared_ptr<Engine> load_engine_from_onnx(const std::string& onnx_file, Precision precision, bool mark_all_output){

    onnx::ModelProto model;
    std::fstream fin(onnx_file, std::ios::binary | std::ios::in);
    if (!model.ParseFromIstream(&fin)) {
        printf("Parse onnx failed.\n");
        return nullptr;
    }

    auto builder = spconv::create_engine_builder();
    auto graph = model.graph();
    
    std::unordered_map<std::string, spconv::ITensor*> tensor_map_by_name;
    for (int i = 0; i < graph.input_size(); ++i) {
        auto name = graph.input(i).name();
        tensor_map_by_name[name] = builder->push_input(name);
    }

    std::vector<spconv::ITensor*> collect_outputs;
    for (int i = 0; i < model.graph().node_size(); ++i) {
        auto& node = model.graph().node(i);
        if (node.op_type() == "SparseConvolution") {

            auto x = tensor_map_by_name[node.input(0)];
            auto weight = get_initializer_data(graph, node.input(1));
            auto bias   = get_initializer_data(graph, node.input(2));
            auto weight_dynamic_ranges_proto = get_attribute(node, "weight_dynamic_ranges");
            auto weight_dynamic_ranges = 
                std::vector<float>(weight_dynamic_ranges_proto.floats().begin(), weight_dynamic_ranges_proto.floats().end());

            auto n = builder->push_sparse_conv(
                node.name(), x, 
                weight.data, weight.shape,
                weight_dynamic_ranges,
                bias.data, bias.shape,
                get_attribute(node, "activation").s(),
                get_attribute_as_intarray(node, "kernel_size"),
                get_attribute_as_intarray(node, "stride"),
                get_attribute_as_intarray(node, "padding"),
                get_attribute_as_intarray(node, "dilation"),
                get_attribute(node, "input_dynamic_range").f(),
                get_attribute(node, "subm").i(),
                get_attribute(node, "output_bound").i(),
                get_attribute(node, "rulebook").s(),
                get_attribute(node, "precision").s() == "int8" ? Precision::Int8 : Precision::Float16,
                get_attribute(node, "output_precision").s() == "int8" ? Precision::Int8 : Precision::Float16,
                node.output(0)
            );

            if(mark_all_output){
                collect_outputs.push_back(n->output(0));
            }
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else if (node.op_type() == "Add" || node.op_type() == "QuantAdd") {
            auto a = tensor_map_by_name[node.input(0)];
            auto b = tensor_map_by_name[node.input(1)];

            auto n = builder->push_add(
                node.name(),
                a, b, 
                get_attribute(node, "input0_dynamic_range").f(),
                get_attribute(node, "input1_dynamic_range").f(),
                node.output(0), 
                get_attribute(node, "precision").s() == "int8" ? Precision::Int8 : Precision::Float16,
                get_attribute(node, "output_precision").s() == "int8" ? Precision::Int8 : Precision::Float16
            );
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else if (node.op_type() == "Relu") {
            auto x = tensor_map_by_name[node.input(0)];
            auto n = builder->push_relu(node.name(), x, node.output(0));
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else if (node.op_type() == "ScatterDense") {
            auto x = tensor_map_by_name[node.input(0)];
            auto input_spatial_shape = get_attribute_as_intarray(node, "input_spatial_shape");
            auto output_shape = get_attribute_as_intarray(node, "output_shape");
            auto format = get_attribute(node, "format").s();
            auto n = builder->push_dense(node.name(), x, format, node.output(0), input_spatial_shape, output_shape);
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else if (node.op_type() == "Reshape") {
            auto x = tensor_map_by_name[node.input(0)];
            auto dims = get_attribute(node, "dims");
            std::vector<int64_t> shape(dims.ints().begin(), dims.ints().end());
            auto n = builder->push_reshape(node.name(), x, shape, node.output(0));
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else if (node.op_type() == "Transpose") {
            auto x = tensor_map_by_name[node.input(0)];
            auto dims = get_attribute(node, "dims");
            std::vector<int64_t> shape(dims.ints().begin(), dims.ints().end());
            auto n = builder->push_transpose(node.name(), x, shape, node.output(0));
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else {
            printf("Unsupport operator [%s]\b", node.op_type().c_str());
            return nullptr;
        }
    }

    for (int i = 0; i < graph.output_size(); ++i) {
        auto name = graph.output(i).name();
        collect_outputs.push_back(tensor_map_by_name[name]);
    }

    for (int i = 0; i < collect_outputs.size(); ++i) {
        builder->push_output(collect_outputs[i]);
    }
    return builder->build(precision);
}
};