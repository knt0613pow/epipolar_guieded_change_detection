import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model.delaunay2D import  Delaunay2D 

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


from model.affinity_layer import InnerpAffinity as Affinity

from pygmtools.classic_solvers import hungarian
from pygmtools.classic_solvers import sinkhorn 
from pygmtools.classic_solvers import sm as SpectralMatching
from pygmtools.classic_solvers import rrwm as RRWM

from model.backbone import FasterRCNN

class GMNET(BaseModel):
    def __init__(self, Weight_path = False):
        super().__init__()
        ## CNN 정의
        self.backbone = FasterRCNN(Weight_path)
        self.affinity_layer = Affinity() ##feature channel
        self.gm_solver = SpectralMatching(max_iter = 5, stop_thresh = 0.5)
        self.sinkhorn = sinkhorn(max_iter = 5, tau = 1/0.2, epsilon = 0.2, log_forward = False)
        self.l2norm = nn.LocalResponseNorm(20, alpha=0.2, beta=0.5, k=0)
        
    def forward(self, image_, **kwargs):
        ref, query = image_
        ref_OD = self.backbone(ref)
        query_OD = self.backbone(query)
        
from torchvision.models.detection import fasterrcnn_resnet50_fpn as FRCNN
import torch
class Pseudo_GMNet(BaseModel):
    def __init__(self, OD_Weight_path = False):
        super().__init__()
        self.backbone = FasterRCNN(OD_Weight_path)
        # self.backbone = torch.jit.script(FRCNN(), output=['roi_head.output', 'box_predictor.output'])
        self.affinity_layer = Affinity(2048)
        self.l2norm = nn.LocalResponseNorm(20, alpha=0.2, beta=0.5, k=0)
        self.multihead_attn = nn.MultiheadAttention(256, 1, batch_first=True)
        
    def forward(self, image_, target = None):
        ref, query = image_
        ref_label , query_label = target



        for b_ref_label, b_query_label in zip(ref_label, query_label):
            b_ref_label['labels'] = torch.zeros_like(b_ref_label['labels'])
            b_query_label['labels'] = torch.zeros_like(b_query_label['labels'])
        # {"image": self.transform(Image.open(i1_path).convert("RGB")), "label": I1_label}
        
        ref_feature, ref_loss, ref_detect = self.backbone(ref, ref_label)
        que_feature, que_loss, que_detect = self.backbone(query, query_label)

        """
        if self.training = True, ref_detect is []
        """

        ref_node_x = [(bat['boxes'][:, 0] + bat['boxes'][:,2]).cpu()/2 for bat in ref_detect]
        ref_node_y = [(bat['boxes'][:, 1] + bat['boxes'][:,3]).cpu()/2 for bat in ref_detect]
        que_node_x = [(bat['boxes'][:, 0] + bat['boxes'][:,2]).cpu()/2 for bat in que_detect]
        que_node_y = [(bat['boxes'][:, 1] + bat['boxes'][:,3]).cpu()/2 for bat in que_detect]

        ref_dt = []
        que_dt = []

        for single_ref_node_x, single_ref_node_y in zip(ref_node_x, ref_node_y):
            single_ref_dt = Delaunay2D()
            [single_ref_dt.addPoint([single_ref_x, single_ref_y]) for single_ref_x , single_ref_y in zip(single_ref_node_x, single_ref_node_y)]
            ref_dt.append(single_ref_dt)

        for single_que_node_x, single_que_node_y in zip(que_node_x, que_node_y):
            single_que_dt = Delaunay2D()
            [single_que_dt.addPoint([single_que_x, single_que_y]) for single_que_x , single_que_y in zip(single_que_node_x, single_que_node_y)]
            que_dt.append(single_que_dt)

        ref_graph = ref_dt.exportGraph()
        que_graph = que_dt.exportGraph()


        breakpoint()


        subdiv = cv2.Subdiv2D((0,0, ref.shape[0], ref.shape[1]));
        
        breakpoint()
        return ref_OD
        