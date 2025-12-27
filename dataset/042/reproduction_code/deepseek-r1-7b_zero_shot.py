model = MyModel().half()
dummy_input = (torch.randn(3, 5), torch.randint(0, 2, (3,)))
torch.onnx.export(model, dummy_input)