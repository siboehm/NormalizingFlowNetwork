from normalizing_flows.InvertedPlanarFlow import InvertedPlanarFlow
from normalizing_flows.InvertedRadialFlow import InvertedRadialFlow
from normalizing_flows.AffineFlow import AffineFlow

FLOWS = {
    'planar': InvertedPlanarFlow,
    'radial': InvertedRadialFlow,
    'affine': AffineFlow
}