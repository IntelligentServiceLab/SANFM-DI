import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch.nn as nn

class SANFM_DI(nn.Module):
    def __init__(self, embed_dim, droprate = 0.5):
        super(SANFM_DI, self).__init__()

        self.i_num = 1536
        self.c_num = 2  #DA:0-767, DM:768-1535, CA:1536, CM:1537
        self.embed_dim = embed_dim  #
        self.att_dim = embed_dim  #
        self.bi_inter_dim = embed_dim  #
        self.droprate = droprate
        self.criterion = nn.BCELoss(weight=None, reduction='mean')
        self.sigmoid = nn.Sigmoid()
        self.hidden_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.fm_linear = nn.Linear((self.i_num + self.c_num), 1, bias=True) #此处维度待修改

        self.dense_embed_DA = nn.Linear(768, self.embed_dim)
        self.dense_embed_DM = nn.Linear(768, self.embed_dim)
        self.hidden_DA = nn.Parameter(torch.randn(self.embed_dim, self.hidden_size))
        self.hidden_DM = nn.Parameter(torch.randn(self.embed_dim, self.hidden_size))
        self.hidden_CA = nn.Parameter(torch.randn(self.embed_dim, self.hidden_size))
        self.hidden_CM = nn.Parameter(torch.randn(self.embed_dim, self.hidden_size))
        self.field_vector_DM = nn.Parameter(torch.randn(1, self.hidden_size))
        self.field_vector_DA = nn.Parameter(torch.randn(1, self.hidden_size))
        self.field_vector_CM = nn.Parameter(torch.randn(1, self.hidden_size))
        self.field_vector_CA = nn.Parameter(torch.randn(1, self.hidden_size))
        
        #以下为selfatt所需参数
        self.query_matrix = nn.Parameter(torch.empty(self.embed_dim, self.att_dim))
        self.key_matrix = nn.Parameter(torch.empty(self.embed_dim, self.att_dim))
        self.value_matrix = nn.Parameter(torch.empty(self.embed_dim, self.att_dim))
        self.softmax = nn.Softmax(dim=-1)	#当nn.Softmax的输入是一个二维张量时，其参数dim = 0，是让列之和为1；dim = 1，是让行之和为1
        #以上为selfatt所需参数

        self.hidden_1 = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.hidden_2 = nn.Linear(self.embed_dim, 1)

        self.bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self._init_weight_()

    def Field_Inter_Pooling(self, inter_DA, inter_DM, inter_CA, inter_CM, batch_size):  
        feature_size_DA = 64    #直接赋值，要用shape获取也行，下同
        feature_size_DM = 64
        feature_size_CA = 1
        feature_size_CM = 1
        hidden_size = self.hidden_size
        #接下来产生隐向量
        inter_DA_hide = inter_DA.unsqueeze(2).repeat(1, 1, self.hidden_size) * self.hidden_DA.unsqueeze(0).repeat(batch_size, 1, 1)
        inter_DM_hide = inter_DM.unsqueeze(2).repeat(1, 1, self.hidden_size) * self.hidden_DM.unsqueeze(0).repeat(batch_size, 1, 1)    #此处hidden_DA和DM是对应的共享的隐向量矩阵，下同，记得在函数首部定义
        inter_CA_hide = inter_CA.unsqueeze(2).repeat(1, 1, self.hidden_size) * self.hidden_CA.unsqueeze(0).repeat(batch_size, 1, 1)
        inter_CM_hide = inter_CM.unsqueeze(2).repeat(1, 1, self.hidden_size) * self.hidden_CM.unsqueeze(0).repeat(batch_size, 1, 1)
        #本域自交互，暂不考虑CA和CM
        inter_DA_hide_out = self.BiInteractionPooling(inter_DA_hide)
        inter_DM_hide_out = self.BiInteractionPooling(inter_DM_hide)

        #跨域交互
        # field_inter_pooling_out = torch.zeros((batch_size, 64))
        Pooling_DA_DM = self.Different_Field_Inter(batch_size, feature_size_DA, feature_size_DM, hidden_size, inter_DA_hide, inter_DM, self.field_vector_DM)    # DA*DM，同理，记得在函数首部定义域隐向量field_vector_XX，目前共计四个
        Pooling_DM_DA = self.Different_Field_Inter(batch_size, feature_size_DM, feature_size_DA, hidden_size, inter_DM_hide, inter_DA, self.field_vector_DA)    # DM*DA
        Pooling_DA_CA = self.Different_Field_Inter(batch_size, feature_size_DA, feature_size_CA, hidden_size, inter_DA_hide, inter_CA, self.field_vector_CA)    # DA*CA
        Pooling_CA_DA = self.Different_Field_Inter(batch_size, feature_size_CA, feature_size_DA, hidden_size, inter_CA_hide, inter_DA, self.field_vector_DA)    # DA*CA
        Pooling_DA_CM = self.Different_Field_Inter(batch_size, feature_size_DA, feature_size_CM, hidden_size, inter_DA_hide, inter_CM, self.field_vector_CM)    # DA*CM
        Pooling_CM_DA = self.Different_Field_Inter(batch_size, feature_size_CM, feature_size_DA, hidden_size, inter_CM_hide, inter_DA, self.field_vector_DA)    # CM*DA
        Pooling_DM_CA = self.Different_Field_Inter(batch_size, feature_size_DM, feature_size_CA, hidden_size, inter_DM_hide, inter_CA, self.field_vector_CA)    # DM*CA
        Pooling_CA_DM = self.Different_Field_Inter(batch_size, feature_size_CA, feature_size_DM, hidden_size, inter_CA_hide, inter_DM, self.field_vector_DM)    # CA*DM
        Pooling_DM_CM = self.Different_Field_Inter(batch_size, feature_size_DM, feature_size_CM, hidden_size, inter_DM_hide, inter_CM, self.field_vector_CM)    # DM*CM
        Pooling_CM_DM = self.Different_Field_Inter(batch_size, feature_size_CM, feature_size_DM, hidden_size, inter_CM_hide, inter_DM, self.field_vector_DM)    # CM*DM
        Pooling_CA_CM = self.Different_Field_Inter(batch_size, feature_size_CA, feature_size_CM, hidden_size, inter_CA_hide, inter_CM, self.field_vector_CM)    # CA*CM
        Pooling_CM_CA = self.Different_Field_Inter(batch_size, feature_size_CM, feature_size_CA, hidden_size, inter_CM_hide, inter_CA, self.field_vector_CA)    # CM*CA
        #叠加获得最终向量，目标是输出为(batch_size, hidden_size)  
        field_inter_pooling_out = inter_DA_hide_out + inter_DM_hide_out + Pooling_DA_DM + Pooling_DM_DA + Pooling_DA_CA + Pooling_CA_DA + Pooling_DA_CM + Pooling_CM_DA + Pooling_DM_CA + Pooling_CA_DM + Pooling_DM_CM + Pooling_CM_DM + Pooling_CA_CM + Pooling_CM_CA
        return field_inter_pooling_out



    def Different_Field_Inter(self, batch_size, feature_size_A, feature_size_B, hidden_size, hidden_matrix_A, inter_matrix_B, field_vector_B):
        field_inter = torch.zeros((batch_size, hidden_size)).to(self.device)
        for k in range(batch_size):
            field_inter_onevector = torch.zeros((1,hidden_size)).to(self.device)
            for i in range(feature_size_A):
                for j in range(feature_size_B):
                    temporary_vector = hidden_matrix_A[k][i] * field_vector_B * inter_matrix_B[k][j] 
                    temporary_vector = temporary_vector.to(self.device)
                    field_inter_onevector += temporary_vector
            field_inter[k] = field_inter_onevector
        return field_inter

    def BiInteractionPooling(self, pairwise_inter):
        inter_part1_sum = torch.sum(pairwise_inter, dim=1)
        inter_part1_sum_square = torch.square(inter_part1_sum)  # square_of_sum

        inter_part2 = pairwise_inter * pairwise_inter
        inter_part2_sum = torch.sum(inter_part2, dim=1)  # sum of square
        bi_inter_out = 0.5 * (inter_part1_sum_square - inter_part2_sum)
        return bi_inter_out


    def _init_weight_(self):
        # deep layers
        nn.init.kaiming_normal_(self.hidden_1.weight)
        nn.init.kaiming_normal_(self.hidden_2.weight)
        # attention part
        nn.init.kaiming_normal_(self.query_matrix)
        nn.init.kaiming_normal_(self.key_matrix)
        nn.init.kaiming_normal_(self.value_matrix)

    def forward(self, batch_data):
        batch_data = batch_data.to(torch.float32)

        batch_size, feature_size = batch_data.shape

        # fm part
        fm_result = self.fm_linear(batch_data)

        # dense embedding preparation for each part
        inter_DA = self.dense_embed_DA(batch_data[:, :768])    #变为低维向量，此处假设是64维，输出为(batch_size, feature_size)
        inter_DM = self.dense_embed_DM(batch_data[:, 768:1536])    #同上
        inter_CA = batch_data[:, 1536].unsqueeze(1)
        inter_CM = batch_data[:, 1537].unsqueeze(1)

        pooling_out = self.Field_Inter_Pooling(inter_DA, inter_DM, inter_CA, inter_CM, batch_size)
        
        # self_attention part
        X = pooling_out
        proj_query = torch.mm(X, self.query_matrix)	#把原先tensor中的数据按照行优先的顺序排成一个一维的数据，然后按照参数组合成其他维度的tensor
        proj_key = torch.mm(X, self.key_matrix)
        proj_value = torch.mm(X, self.value_matrix)

        S = torch.mm(proj_query, proj_key.T)
        attention_map = self.softmax(S) #这里只是q*k

        # Self-Attention Map
        value_weight = proj_value[:,None] * attention_map.T[:,:,None]
        value_weight_sum = value_weight.sum(dim=0)  #此处认为是已经加权过了的原数据值，可以直接用于下一步。
        #以上为自注意力
        
        #MLP part
        mlp_hidden_1 = F.relu(self.bn(self.hidden_1(value_weight_sum))) 
        mlp_hidden_2 = F.dropout(mlp_hidden_1, training=self.training, p=self.droprate)
        mlp_out = self.hidden_2(mlp_hidden_2)
        final_out = fm_result + mlp_out
        final_sig_out = self.sigmoid(final_out)
        final_sig_out_squeeze = final_sig_out.squeeze()
        return final_sig_out_squeeze  

    def loss(self, batch_input, batch_label):
        pred = self.forward(batch_input)
        pred = pred.to(torch.float32)
        batch_label = batch_label.to(torch.float32).squeeze()
        loss1 = self.criterion(pred, batch_label)
        return loss1


def train(model, train_loader, optimizer, epoch, device):
    model.train()
    # global loss_train
    avg_loss = 0.0

    for i, data in enumerate(train_loader):
        batch_input, batch_label = data
        batch_input = batch_input.to(device)
        batch_label = batch_label.to(device)
        optimizer.zero_grad()
        loss2 = model.loss(batch_input, batch_label)
        loss2.backward(retain_graph = True)
        optimizer.step()

        avg_loss += loss2.item()

        if (i + 1) % 10 == 0:
            print('%s Training: [%d epoch, %3d batch] loss: %.5f' % (
                datetime.now(), epoch, i + 1, avg_loss / 10))
            # loss_train = avg_loss
            avg_loss = 0.0
    return 0


def tst(model, test_loader, device):
    model.eval()
    criterion = nn.BCELoss(reduction='mean')
    LOSS = []
    AUC = []

    for test_input, test_label in test_loader:
        test_input = test_input.to(device)
        test_label = test_label.to(device)
        pred = model(test_input)
        pred = pred.to(torch.float32)
        test_label = test_label.squeeze().to(torch.float32)
        loss_value = criterion(pred, test_label)
        auc_value = roc_auc_score(test_label.detach().tolist(), pred.detach().tolist())
        LOSS.append(loss_value.detach())
        AUC.append(auc_value)
    loss = np.mean(LOSS)
    auc = np.mean(AUC)
    return loss, auc

def main():
    #cuda implement
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #load data
    data = pd.read_csv('./input_bert_data.csv')
    
    sparse_features = ['1538', '1539']
    dense_features = [str(i) for i in range(2, 1538)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    # target = ['1']  # 数据读取完毕
    # 1.Label Encoding for sparse features,and do simple Transformation for dense features 对稀疏特征进行标签编码，对密集特征进行简单变换
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    #2.格式化数据
    data_np = np.array(data)
    data_np1, data_np2 = np.split(data_np, [1], axis=1)  # 拆分成两个array，tensor化后可以直接接tensordata
    x_train, x_test, y_train, y_test = train_test_split(data_np2, data_np1, test_size=0.4, random_state=1)

    x_train_tensor = torch.tensor(x_train)
    x_test_tensor = torch.tensor(x_test)
    y_train_tensor = torch.tensor(y_train)
    y_test_tensor = torch.tensor(y_test)

    train_tensor_set = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=train_tensor_set, batch_size=100, shuffle=True)
    test_tensor_set = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = torch.utils.data.DataLoader(dataset=test_tensor_set, batch_size=100, shuffle=True)

    #实例化模型

    sanfm_di_obj = SANFM_DI(embed_dim=64)
    sanfm_di_obj = sanfm_di_obj.to(device)
    optimizer = torch.optim.Adam(sanfm_di_obj.parameters(), lr=0.001)
    loss_train = np.inf
    LOSS_total = np.inf
    # AUC_total = np.inf
    endure_count = 0

    for epoch in range(250):
        #train
        train(sanfm_di_obj, train_loader, optimizer, epoch, device)
        #test
        loss_train, AUC = tst(sanfm_di_obj, test_loader)

        if LOSS_total > loss_train:
            LOSS_total = loss_train
            # AUC_total = AUC
            endure_count = 0
        else:
            endure_count += 1

        print("<Test> LOSS: %.5f AUC: %.5f" % (loss_train, AUC))

        if endure_count > 30: 
            break
    LOSS, AUC = tst(sanfm_di_obj, test_loader, device)


    print('The best LOSS: %.5f AUC: %.5f' % (LOSS, AUC))

if __name__ == "__main__":
    main()
