from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .second_head import SECONDHead
from .voxelrcnn_head import VoxelRCNNHead
from .roi_head_template import RoIHeadTemplate

from .partA2_basd_head import PartA2FCBasdHead
from .voxelrcnn_based_head import VoxelRCNNBasedHead


__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PartA2FCBasdHead': PartA2FCBasdHead,
    'PVRCNNHead': PVRCNNHead,
    'SECONDHead': SECONDHead,
    'PointRCNNHead': PointRCNNHead,
    'VoxelRCNNHead': VoxelRCNNHead,
    'VoxelRCNNBasedHead': VoxelRCNNBasedHead
}
