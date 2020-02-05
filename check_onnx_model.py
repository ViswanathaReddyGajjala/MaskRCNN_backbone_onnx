import numpy as np
import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend
from backbone_with_fpn import resnet_fpn_backbone


model = onnx.load("backbone_fpn.onnx")

backbone = resnet_fpn_backbone('resnet18', pretrained= True)

torch_model = backbone
#torch_model = torch_model.cuda()
torch_model.eval()

x = torch.randn(1, 3, 624, 624, requires_grad=True)


prepared_backend = onnx_caffe2_backend.prepare(model)

# run the model in Caffe2
W = {model.graph.input[0].name: x.data.numpy()}

# Run the Caffe2 net:
onnx_out = prepared_backend.run(W)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Verify the numerical correctness upto 3 decimal places
for i in range(len(torch_out)):
    print (torch_out[i].shape, onnx_out[i].shape)
    np.testing.assert_almost_equal(to_numpy(torch_out[i]), onnx_out[i], decimal=3)
    #np.testing.assert_almost_equal(to_numpy(torch_out[i]).reshape(c, h, w), c2_out[i], decimal=3)
print("Exported model has been executed on Caffe2 backend, and the result looks good!")

