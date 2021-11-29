import time
import copy
import numpy as np
#import torch
import logging

logger = logging.getLogger("logger")
import sklearn.metrics.pairwise as smp

import bb_tsss_helper as ts


# ASF defense (Area Similarity FoolsGold) is a new defense building on FoolsGold featuring TS-SS
# as the similarity algorithm used in comparing user gradient updates to detect and mitigate sybils
# seen here: https://www.analyticsvidhya.com/blog/2021/06/nlp-answer-retrieval-from-document-using-ts-ss-similarity-python/
# and here: https://www.computer.org/csdl/proceedings-article/bigdataservice/2016/2251a142/12OmNweBUID

class ASF(object):
    def __init__(self, use_memory=False):
        self.memory = None
        self.memory_dict = dict()
        self.wv_history = []
        self.use_memory = use_memory

    def aggregate_gradients(self, client_grads, names):
        cur_time = time.time()
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()

        self.memory = np.zeros((num_clients, grad_len))
        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]] += grads[i]
            else:
                self.memory_dict[names[i]] = copy.deepcopy(grads[i])
            self.memory[i] = self.memory_dict[names[i]]
        # self.memory += grads

        if self.use_memory:
            wv, alpha = self.asf(self.memory)  # Use FG
        else:
            wv, alpha = self.asf(grads)  # Use FG
        logger.info(f'[asf agg] wv: {wv}')
        self.wv_history.append(wv)

        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(
                len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i].cpu()
            temp = temp / len(client_grads)
            agg_grads.append(temp)
        print('model aggregation took {}s'.format(time.time() - cur_time))
        return agg_grads, wv, alpha

    def asf(self, grads):
        """
        :param grads:
        :return: compute similatiry and return weightings
        """
        n_clients = grads.shape[0]

        # ------------------------------------------------------------
        #    PRIMARY ALGORITHM:
        #    TS-SS Triangle Area Similarity - Sector Area Similarity (primary)
        #
        #    OPTIONAL TESTING ALGORITHMS
        #    Euclidean Distance
        #    Manhattan Distance
        #
        #    prc constant is a hyperparameter for tuning learning rates and weight distribution
        #    of participants pardoning
        #
        #    Note: Distance calculations, Normalized to [-1, 1], from:
        #    (https://www.codegrepper.com/code-examples/python/how+to+scale+an+array+between+two+values+python)

        #    1.  Euclidean Normalized
        # distance_calc = smp.euclidean_distances(grads)
        # normalized = 2.*(distance_calc - np.min(distance_calc))/np.ptp(distance_calc)-1
        # sm = normalized - np.eye(n_clients)
        # prc = 0.1 # adjust value to improve results

        #    2.  Manhattan Normalized
        # distance_calc = smp.manhattan_distances(grads)
        # normalized = 2.*(distance_calc - np.min(distance_calc))/np.ptp(distance_calc)-1
        # sm = normalized - np.eye(n_clients)
        # prc = 0.1 # adjust value to improve results

        #    3.  TS-SS Triangle Area Similarity - Sector Area Similarity
        v = torch.tensor(grads)

        # TS-SS normalized
        distance_calc = ts.ts_ss_(v).numpy()
        normalized = 2. * (distance_calc - np.min(distance_calc)) / np.ptp(distance_calc) - 1
        sm = normalized - np.eye(n_clients)
        prc = 0.05

        # To run unnormalized version of TS-SS (not recommended)
        # sm = ts.ts_ss_(v).numpy() - np.eye(n_clients)

        # ------------------------------------------------------------

        # Testing/Debugging code
        #        print('TEST TS SS')
        #        print('TEST TS SS')
        #        print('TEST TS SS')
        #        print('TEST TS SS')

        #        v = torch.tensor(grads)
        #        sm = ts_ss_(v).numpy()

        #        print('TEST TS SS 1')
        #        v = torch.tensor(grads)
        #        print('TEST TS SS 2')
        #        sm = ts_ss_(v)
        #        print('TEST TS SS 3')
        #        print(sm)
        #        print('TEST TS SS 4')
        #        print(ts_ss_(v))
        #        print('TEST TS SS 5')

        #        cs_list = list()
        #        for i, x in enumerate(grads):
        #            for j, y in enumerate(grads):
        #                v = torch.tensor([x, y])
        #                res = ts_ss_(v)
        #                cs_list.append(res)

        # vec1 = [1.0,2.0]
        # vec2 = [2.0,4.0]
        # v = torch.tensor([vec1, vec2])
        # print(ts_ss_(v))

        # for g1, g2 in enumerate(grads):
        #    print(ts_ss_(g2))

        # ------------------------------------------------------------

        # failed efforts:
        # ed_normVector = np.linalg.norm(edtest)
        # ed_normalized = edtest/ed_normVector
        # ed_transformed = np.linalg.norm(ed_normalized)  # = 1.0
        # norm1 = F.normalize(edtest)
        # norm2 = F.normalize(edtest, dim=2)
        # norm3 = F.normalize(edtest, dim=-1)
        # ed_normL = np.linalg.norm(edtest, axis=0) -
        # ed_normalizedL = edtest/ednormL - fails
        # ------------------------------------------------------------

        maxsm = np.max(sm, axis=1)

        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxsm[i] < maxsm[j]:
                    sm[i][j] = sm[i][j] * maxsm[i] / maxsm[j] * prc
        wv = 1 - (np.max(sm, axis=1))

        wv[wv > 1] = 1
        wv[wv < 0] = 0

        alpha = np.max(sm, axis=1)

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        # wv is the weight
        return wv, alpha