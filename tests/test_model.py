import torch

from littera.model import LineModel, ModelConfig, parameter_count


def test_shapes_for_every_backbone_and_context():
    images = torch.rand(2, 1, 32, 120)
    widths = torch.tensor([120, 88])
    for backbone, context in [("resnet", "bilstm"), ("resnet", "local"), ("vgsl", "kraken"), ("vgsl", "local")]:
        config = ModelConfig(n_labels=7, backbone=backbone, context=context, height=32, channels=[8, 8, 16, 16], hidden=16, layers=1)
        model = LineModel(config).eval()
        logits, boundary, lengths = model(images, widths)
        assert logits.shape == (2, 30, 7) and boundary.shape == (2, 30)
        assert lengths.tolist() == [30, 22]


def test_full_size_parameter_counts():
    counts = {(b, c): parameter_count(LineModel(ModelConfig(n_labels=200, backbone=b, context=c))) for b, c in
              [("resnet", "bilstm"), ("resnet", "local"), ("vgsl", "kraken")]}
    assert 6.0e6 < counts[("resnet", "bilstm")] < 9.0e6
    assert 5.5e6 < counts[("resnet", "local")] < 9.0e6
    assert 3.0e6 < counts[("vgsl", "kraken")] < 4.5e6
