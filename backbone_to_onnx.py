from backbone_with_fpn import resnet_fpn_backbone
import torch

backbone = resnet_fpn_backbone('resnet18', pretrained= True)

torch_model = backbone
#torch_model = torch_model.cuda()
torch_model.eval()

x = torch.randn(1, 3, 624, 624, requires_grad=True)
torch_out = torch_model(x)

print ([out.shape for out in torch_out])


# Export the model
torch.onnx.export(torch_model,               
                  x,                        
                  "backbone_fpn.onnx",   
                  input_names = ["input"],
                  output_names = ['output1', 'output2', 'output3', 'output4', 'output5'],
                  export_params=True, keep_initializers_as_inputs=True
                 )

