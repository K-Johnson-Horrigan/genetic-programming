import numpy as np


class Network:

    def __init__(self, nodes, links):

        # List of nodes as [x,y]
        self.nodes = np.array(nodes)

        # List of all links as [node_1, node_2]
        self.links_list = tuple(tuple(l) for l in links)

        # Adj matrix of all links_adj
        self.links_adj = np.zeros((len(nodes), len(nodes)))
        for i,j in links:
            # i,j = sorted((i,j))
            self.links_adj[i,j] = 1
            self.links_adj[j,i] = 1

        self.update_dists()





    def update_dists(self):
        """Updates the distances between links using the index of each link in the links_list"""
        self.dists = -np.ones((len(self.links_list), len(self.links_list)))

        for s1 in range(len(self.links_list)):
            for s2 in range(len(self.links_list)):

                l1 = self.links_list[s1]
                l2 = self.links_list[s2]

                d00 = np.linalg.norm(l1[0] - l2[0])
                d01 = np.linalg.norm(l1[0] - l2[1])
                d10 = np.linalg.norm(l1[1] - l2[0])
                d11 = np.linalg.norm(l1[1] - l2[1])

                dist = min([d00,d01,d10,d11])

                self.dists[s1,s2] = dist
                self.dists[s2,s1] = dist







if __name__ == '__main__':

    a = Network([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ],[
        [0, 1],
        [0, 2],
        [3, 1],
        [3, 2],
    ])

    pass

    # b = Network()




