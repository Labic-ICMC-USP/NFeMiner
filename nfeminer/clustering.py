import copy
import networkx as nx
import numpy as np
import random
from collections import Counter
from itertools import chain
from .similarity_graph import *
import itertools

## community_algorithms:

from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community import asyn_lpa_communities
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import k_clique_communities
from networkx.algorithms.community import kernighan_lin_bisection
from typing import Dict, Any, List, Tuple, Optional, Union
from types import SimpleNamespace

def direct_nearest_neighbors(G: nx.Graph,
                             node_id: Any,
                             k: int,
                             weight: Optional[str] = None,
                             maximize: bool = False) -> List[Tuple[Any, float]]:
    """Return the direct k nearest neighbors of a node using an edge attribute.

    For each neighbor of `node_id`, this function reads the edge attribute
    named by `weight` and uses it as the similarity/score. If the attribute
    is missing on an edge, `float("inf")` is used as a fallback (which means
    that when `maximize` is False such neighbors will appear last; when
    `maximize` is True they may appear first).

    Args:
        G (nx.Graph): Graph containing the node and edges.
        node_id: Identifier of the node whose neighbors are queried.
        k (int): Number of neighbors to return (top-k).
        weight (str | None): Edge attribute name used as the score. If None,
            the code will try to `get(None)` and fall back to `inf` for missing values.
        maximize (bool): If True, larger scores are considered better (sort
            in descending order). If False, smaller scores are considered
            better (ascending order).

    Returns:
        List[Tuple[Any, float]]: List of (neighbor_id, score) for the top-k neighbors.
    """
    neighbors: List[Tuple[Any, float]] = []
    for neighbor in G.neighbors(node_id):
        edge_data = G.get_edge_data(node_id, neighbor) or {}
        sim = edge_data.get(weight, float("inf"))
        neighbors.append((neighbor, sim))
    neighbors.sort(key=lambda x: x[1], reverse=maximize)
    return neighbors[:k]

def semi_supervised_label_propagation(G: nx.Graph,
                                      initial_labels: Dict[Any, Any],
                                      max_iter: int = 100,
                                      tol: float = 1e-3,
                                      seed: int = None,
                                      UNKOWN_LABEL = -1,
                                      inertia: float = 0.5) -> Dict[Any, Any]:
    """Perform semi-supervised label propagation on a graph.

    This function propagates labels from a set of initially labeled nodes to
    the rest of the nodes using a simple iterative weighted-vote scheme.
    At each iteration nodes are visited in random order and adopt the label
    that maximizes the sum of neighbor weights for each label. Previously
    fixed (initial) labels are never changed. Optionally, an inertia term
    biases nodes toward their current label.

    The method stops when either the maximum number of iterations is reached
    or when the relative fraction of changed labels in an iteration is below
    `tol`.

    Args:
        G (nx.Graph): Input graph. Edge weights are read from the
            `'weight'` attribute; when absent weight defaults to 1.0.
        initial_labels (Dict[Any, Any]): Mapping {node: label} of nodes that
            are fixed / known at the start (these labels are not modified).
        max_iter (int): Maximum number of propagation iterations. Default: 100.
        tol (float): Convergence tolerance. If (changes / n_nodes) < tol the
            algorithm stops early. Default: 1e-3.
        seed (int | None): Seed for the internal random generator used to
            shuffle node visit order and to break ties. Default: None.
        UNKOWN_LABEL: Value that represents the unknown/unlabeled state.
            Default: -1. (Note: parameter name kept as in original.)
        inertia (float): Value in [0, 1] that biases a node toward keeping
            its current label. The current label's weight is combined as
            `(1 - inertia) * neighbor_weight + inertia`. Default: 0.5.

    Returns:
        Dict[Any, Any]: Mapping {node: label} with propagated labels for all nodes.
    """
    rng = random.Random(seed)
    labels = {node: UNKOWN_LABEL for node in G.nodes()}
    labels.update(initial_labels)
    fixed = set(initial_labels.keys())
    n = G.number_of_nodes()
    for _ in range(max_iter):
        changes = 0
        nodes = list(G.nodes())
        rng.shuffle(nodes)
        for u in nodes:
            if u in fixed:
                continue
            weights = Counter()
            for v in G.neighbors(u):
                lbl = labels.get(v)
                if lbl == UNKOWN_LABEL:
                    continue
                w = G[u][v].get('weight', 1.0)
                weights[lbl] += w
            if not weights:
                continue
            current = labels[u]
            if current != UNKOWN_LABEL:
                weights[current] = (1 - inertia) * weights.get(current, 0) + inertia
            max_w = max(weights.values())
            best = [lbl for lbl, w in weights.items() if w == max_w]
            new_lbl = rng.choice(best)
            if labels[u] != new_lbl:
                labels[u] = new_lbl
                changes += 1
        if changes / n < tol:
            break
    return labels

class CGraph():
    def __init__(self, G: nx.Graph, name: str = None):
        """Initialize a CGraph.

        This class wraps a NetworkX graph and manages node labels and
        multiple label sets produced by clustering operations.

        Args:
            G (nx.Graph): Similarity graph to wrap.
            name (str, optional): Optional graph name.
        """
        super().__init__()
        self.G = G
        self.name = name
        self.multilabels = []  # stores multiple sets of labels
        self.next_label = 0

        # Ensure nodes have a 'label' attribute; if none of the nodes have it,
        # initialize all node labels to -1.
        has_labels = False
        for node_id in G.nodes():
            if 'label' in G.nodes[node_id]:
                has_labels = True
                break
        if not has_labels:
            nx.set_node_attributes(self.G, -1, "label")

    def __len__(self):
        """Return number of nodes in the wrapped graph."""
        return len(self.G)

    def __getitem__(self, index):
        """Return node attribute dict for node `index` (delegates to nx)."""
        return self.G.nodes[index]

    def copy(self, data: bool = True):
        """Return a deep copy of this CGraph.

        Args:
            data (bool): If True, copy node attributes into the new graph.

        Returns:
            CGraph: A deep-copied CGraph instance.
        """
        graph = CGraph(copy.deepcopy(self.G))
        if data:
            node_attrs = {n: d.copy() for n, d in self.G.nodes(data=True)}
            nx.set_node_attributes(graph.G, node_attrs)
        return graph

    @property
    def labels(self):
        """Return a mapping {node_id: label} from the wrapped graph."""
        return nx.get_node_attributes(self.G, 'label')

    def nodes(self, **kwargs):
        """Return the nodes view of the wrapped graph.

        Accepts the same keyword arguments as `networkx.Graph.nodes(...)`.

        Returns:
            NodeView: NetworkX nodes view (possibly filtered by kwargs).
        """
        return self.G.nodes(**kwargs)

    def degree(self, index, **kwargs):
        """Return the degree or degree view for a node or nodes.

        Args:
            index: Node or collection of nodes (same semantics as networkx).
            **kwargs: Extra keyword arguments forwarded to `self.G.degree`.

        Returns:
            int or DegreeView: Degree count or view depending on index.
        """
        return self.G.degree(index, **kwargs)

    def subgraph(self, nodes):
        """Return a CGraph wrapping the subgraph induced by `nodes`.

        Args:
            nodes (iterable): Iterable of node IDs to induce the subgraph.

        Returns:
            CGraph: New CGraph wrapping the induced subgraph.
        """
        return CGraph(self.G.subgraph(nodes))

    def nxcluster(self, method: str, **params):
        """Apply a NetworkX clustering/community detection method to self.G.

        The `method` specifies which NetworkX community algorithm to run and
        `params` are forwarded to the chosen function.

        Supported methods and the NetworkX functions they use:
            - "label_propagation": asyn_lpa_communities(self.G, **params)
                params: weight (str | None), seed (int | None)
            - "girvan_newman": girvan_newman(self.G, **params)
                params: most_valuable_edge (callable | None)
            - "bisection": kernighan_lin_bisection(self.G, **params)
                params: partition (tuple[list, list] | None), max_iter (int), weight (str), seed (int)
            - "clique": k_clique_communities(self.G, **params)
                params: k (int)
            - "greedy": greedy_modularity_communities(self.G, **params)
                params: weight (str | None)

        After computing communities, this method:
            * Converts communities into a dict {label: [nodes...]}
            * Sets node attribute 'label' for nodes according to the detected community
            * Appends the labels dict into self.multilabels
            * Updates self.next_label to the next available label id

        Args:
            method (str): Name of the clustering method to use.
            **params: Parameters forwarded to the chosen NetworkX function.

        Returns:
            dict[int, list]: Mapping cluster_label -> list of node IDs.

        Raises:
            ValueError: If the method name is not supported.
        """
        if method == 'label_propagation':
            communities = list(asyn_lpa_communities(self.G, **params))
        elif method == 'girvan_newman':
            communities = list(girvan_newman(self.G, **params))
        elif method == 'bisection':
            communities = list(kernighan_lin_bisection(self.G, **params))
        elif method == 'clique':
            communities = list(k_clique_communities(self.G, **params))
        elif method == 'greedy':
            communities = list(greedy_modularity_communities(self.G, **params))
        else:
            raise ValueError(f"Method [{method}] not allowed")

        labels = {label: list(nodelist) for label, nodelist in enumerate(communities)}
        nodelabels = {}
        self.next_label = 0
        for label, nodelist in labels.items():
            for nodeid in nodelist:
                nodelabels[nodeid] = label
            self.next_label = max(self.next_label, label)  # highest label from clustering
        self.next_label += 1  # set to next available label
        nx.set_node_attributes(self.G, nodelabels, 'label')
        self.multilabels.append(labels)
        return labels

    def sscluster(self, method: str, **params):
        """Apply a semi-supervised clustering/label-propagation method.

        Supported `method` values:
            - "label_propagation": uses existing node labels in the graph as
              initial labeled nodes and runs a semi-supervised label propagation.

        For 'label_propagation', the graph's node labels are read (labels != -1
        are treated as ground truth). The result mapping {node: label} is set
        as the node attribute 'label' and appended to self.multilabels.

        Args:
            method (str): Name of the semi-supervised method to run.
            **params: Parameters for the chosen semi-supervised algorithm.

        Raises:
            ValueError: If the method name is not supported.
        """
        if method == 'label_propagation':
            init = nx.get_node_attributes(self.G, 'label')
            initial_labels = {node: lab for node, lab in init.items() if lab != -1}
            result = semi_supervised_label_propagation(
                self.G, initial_labels,
                max_iter=params.get('max_iter', 100),
                tol=params.get('tol', 1e-3),
                seed=params.get('seed', None),
                inertia=params.get('inertia', 0.5)
            )
            # Update node attributes and multilabels
            nx.set_node_attributes(self.G, result, 'label')
            self.multilabels.append(result)
        else:
            raise ValueError(f"SS method [{method}] not allowed")
    
    def core(self, topk: int | None = None, method: str = 'degree', weight=None, return_scores=False):
        """Select the most representative nodes according to a centrality/core metric.

        Args:
            topk (int | None): Number of top nodes to return. If None, return all nodes ordered.
            method (str): Centrality method. One of:
                'degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank', 'corenumber'
            weight (str | None): Edge attribute to use as weight. Default is None.
            return_scores (bool): If True, returns a list of (node_id, score) tuples.
                                  If False, returns only node IDs.

        Returns:
            list: If return_scores is False: list[node_id]. If True: list[(node_id, score)].

        Raises:
            ValueError: If the method is not supported or incompatible with weight.
        """
        if method == 'degree':
            # representatives = nx.degree_centrality(self.G)
            representatives = dict(self.G.degree(weight=weight))
        elif method == 'betweenness':
            representatives = nx.betweenness_centrality(self.G, normalized=True, weight=weight)
        elif method == 'closeness':
            representatives = nx.closeness_centrality(self.G, distance=weight)
        elif method == 'eigenvector':
            representatives = nx.eigenvector_centrality(self.G, max_iter=1000, weight=weight)
        elif method == 'pagerank':
            representatives = nx.pagerank(self.G, alpha=0.85, weight=weight)
        elif method == 'corenumber':
            if weight is not None:
                raise ValueError(f"core number is not supported with weight!=None. weight = [{weight}]")
            representatives = nx.core_number(self.G)
        else:
            raise ValueError(f"Method [{method}] not allowed")

        sorted_items = sorted(representatives.items(), key=lambda x: x[1], reverse=True)[:topk]
        if return_scores:
            return list(sorted_items)
        return [x[0] for x in sorted_items]

    def dismember(self):
        """Split the graph into multiple CGraph objects grouped by current node labels.

        Returns:
            dict[label, CGraph]: Mapping each label to a CGraph containing the nodes
                                 that currently have that label.

        Notes:
            - Restores the original 'label' attributes on the main graph (no change).
            - Updates self.next_label to the maximum found label.
        """
        labels = nx.get_node_attributes(self.G, 'label')
        label2nodes = {}  # map label -> list[node_id]
        for node_id, label in labels.items():
            if label in label2nodes:
                label2nodes[label].append(node_id)
            else:
                label2nodes[label] = [node_id]
        label2cgraph = {label: CGraph(self.G.subgraph(node_list)) for label, node_list in label2nodes.items()}
        nx.set_node_attributes(self.G, labels, 'label')  # restore the labels to the graph
        self.next_label = max(list(label2cgraph.keys()))
        return label2cgraph

    def unknown(self):
        """Return a CGraph containing nodes with unknown label (-1).

        Returns:
            CGraph: Subgraph containing only nodes whose label == -1.
        """
        nodes = nx.get_node_attributes(self.G, 'label')
        unknown_nodes = [node_id for node_id, label in nodes.items() if label == -1]
        nx.set_node_attributes(self.G, nodes, 'label')  # restore labels to the graph
        return CGraph(self.G.subgraph(unknown_nodes))

    def __lshift__(self, other):
        """Assign labels from a subgraph (or a single label) back to this graph.

        Two usage patterns:
            * graph << subgraph (other is CGraph): copies and remaps subgraph labels
              into the parent graph, ensuring label ids do not collide.
            * graph << label (other is int): set the entire graph to a single label.

        Args:
            other (CGraph | int): If CGraph, its node labels are remapped and
                                  applied to self. If int, assign that label to all nodes.

        Returns:
            CGraph: self (after modification).
        """
        if isinstance(other, CGraph):
            # Case: graph << subgraph
            subgraph_nodes_labels = other.G.nodes(data='label')
            subgraph_unique_labels = np.unique([label for node_id, label in subgraph_nodes_labels])
            subgraph_label_remapping = {
                int(label): int(self.next_label + i if label != -1 else -1) for i, label in enumerate(subgraph_unique_labels)
            }
            label_setter = {
                node_id: subgraph_label_remapping[label] for node_id, label in subgraph_nodes_labels
            }
            self.next_label = self.next_label + len(subgraph_unique_labels)
            nx.set_node_attributes(self.G, label_setter, 'label')
            return self
        else:
            # Case: graph << label
            self.next_label = other + 1
            nx.set_node_attributes(self.G, other, 'label')
            return self

    def set_labels(self, node_ids: list[str], label):
        """Set `label` for the given list of node IDs.

        Args:
            node_ids (list[str]): List of node IDs to set the label for.
            label: Label to assign to the provided nodes.
        """
        source_attrs = {
            n: label for n in self.G.nodes() if (n in node_ids)
        }
        nx.set_node_attributes(self.G, source_attrs, 'label')

    def set_attr_from(self, graph, data_attrs: list[str]):
        """Copy specific node attributes from `graph` into a copy of self.

        This method performs a total assignment of attributes (no remapping).
        It returns a new CGraph (a copy of self) with requested attributes copied
        from the provided `graph`. If 'label' is among attributes, the next_label
        of the new graph is also updated from the source graph.

        Args:
            graph (CGraph): Source graph from which attributes are copied.
            data_attrs (list[str]): List of node attribute names to copy.

        Returns:
            CGraph: New CGraph formed by the assignment of attributes.
        """
        new_cgraph = self.copy()

        for attr in data_attrs:
            if attr == 'label':
                new_cgraph.next_label = graph.next_label

            # Extract only attributes of nodes that exist in both graphs
            source_attrs = {
                n: d[attr] for n, d in graph.G.nodes(data=True)
                if attr in d and n in new_cgraph.G.nodes
            }
            nx.set_node_attributes(new_cgraph.G, source_attrs, attr)

        return new_cgraph

    @classmethod
    def pattern(cgraph_cls, my_graph, pattern_graph, method: str, from_my_graph=True):
        """Modify a graph based on the structure of another pattern graph.

        Supported `method` values:
            - 'cut_edges': Return a graph C where C has the same nodes as A
                 and only the edges that are present in both A and B
                 (intersection of edges).
                 After: len(A) == len(C)
            - 'cut_nodes': Return a graph C that contains the nodes in A that are also present in B.
                 Edges and nodes not in the intersection are removed, so len(A) != len(B) generally.

        Args:
            cgraph_cls: Class (CGraph) -- kept for signature compatibility.
            my_graph (CGraph): Primary graph (A) used in the operation.
            pattern_graph (CGraph): Pattern graph (B) used to cut A.
            method (str): 'cut_edges' or 'cut_nodes'.
            from_my_graph (bool): If True, produce the new graph from `my_graph`;
                                  otherwise, produce it from `pattern_graph`.

        Returns:
            CGraph: Resulting graph after applying the pattern operation.

        Raises:
            ValueError: If the method is not supported.
        """
        if method == 'cut_edges':
            pattern_edges = set(list(pattern_graph.G.edges))
            self_edges = set(list(my_graph.G.edges))
            intersection_edges = self_edges & pattern_edges

            new_cgraph = (my_graph if from_my_graph else pattern_graph).copy()
            new_cgraph.G.remove_edges_from(list(self_edges - intersection_edges))
            return new_cgraph
        elif method == 'cut_nodes':
            pattern_nodes = set(list(pattern_graph.G.nodes))
            self_nodes = set(list(my_graph.G.nodes))
            intersection_nodes = self_nodes & pattern_nodes

            new_cgraph = (my_graph if from_my_graph else pattern_graph).copy()
            new_cgraph.G.remove_nodes_from(list(self_nodes - intersection_nodes))
            return new_cgraph
        else:
            raise ValueError(f"Method [{method}] not allowed")

    def direct_k_nearest_neighbors(self, node_id, k: int, weight=None, maximize=True):
        """Wrapper to compute k nearest neighbors for a node using provided helper.

        Args:
            node_id: Node identifier for which to find neighbors.
            k (int): Number of nearest neighbors to return.
            weight: Edge attribute used as weight or distance (forwarded).
            maximize (bool): If True, treat higher values as better (forwarded).

        Returns:
            list: Result of direct_nearest_neighbors(self.G, node_id, k, ...)
        """
        return direct_nearest_neighbors(self.G, node_id, k, weight=weight, maximize=maximize)

















class Analyser:
    """
    A class for analyzing and manipulating CGraph cluster structures.
    
    Provides methods for deep clique clustering, cluster representative 
    visualization, graph dismembering, and group merging operations.
    """
    
    def __init__(self):
        """Initialize an Analyser object."""
        pass
    
    def deep_clique(self, G: CGraph, max_iter: int = 10, step: int = 3) -> CGraph:
        """
        Apply recursive clique clustering to the graph.
        
        Args:
            G (CGraph): Input graph.
            max_iter (int, optional): Maximum number of iterations. Defaults to 10.
            step  (int, optional): Step between iterations
        
        Returns:
            CGraph: The clustered graph.
        """
        return self.__deep_clique(G, max_iter=max_iter, step=step)

    def __deep_clique(self, G: CGraph, max_iter: int = 10, step: int = 3) -> CGraph:
        """
        Internal recursive method for performing clique clustering.
        
        Args:
            G (CGraph): Input graph.
            max_iter (int, optional): Maximum number of iterations. Defaults to 10.
            step  (int, optional): Step between iterations

        Returns:
            CGraph: The clustered graph after recursive clique clustering.
        """
        if max_iter <= 1:
            return G
        
        G.nxcluster('clique', k=max_iter)
        subgraphs = G.dismember()
        
        try:
            unknown_subgraph = subgraphs[-1]
        except:
            return G
        
        sub_G = self.__deep_clique(unknown_subgraph, max_iter=max_iter-step)
        G << sub_G  # Assigns labels from the subgraph to the larger graph
        return G
     
    def show_representatives(
        self,
        G: CGraph,
        n_objs: int = 1,
        weight: str = None,
        core_method: str = 'degree',
        only_unknown: bool = False,
        not_display_unknown: bool = False,
        filter_node_ids: list[str] = None,
        show_node_id: bool = False,
        TEXT_COLUMN:str='produto_detalhado',
        ID_GTIN_COLUMN:str='gtin'
    ):
        """
        Display representative nodes of each cluster in the terminal.

        Args:
            G (CGraph): The input graph.
            n_objs (int, optional): Number of representative nodes to display per cluster. Defaults to 1.
            weight (str, optional): Attribute label used in representative calculation.
            core_method (str, optional): GCluster method used to compute representatives. Defaults to 'degree'.
            only_unknown (bool, optional): If True, only display cluster -1. Assumes all clustered nodes are correct. Defaults to False.
            not_display_unknown (bool, optional): If True, hide unknown cluster (-1). Defaults to False.
            filter_node_ids (list[str], optional): Show clusters only if they contain at least one of these node_ids. Defaults to None.
            show_node_id (bool, optional): If True, also display the node_id. Defaults to False.
            TEXT_COLUMN (str): Defines the column of string to be show. By default it's 'produto_detalhado'.
            ID_GTIN_COLUMN (str): Defines the column of gtin to be show. By default it's 'gtin'.
        """
        subgraphs = G.dismember()
        if -1 in subgraphs:
            print(f"Clustered nodes: {len(G) - len(subgraphs[-1])} / {len(G)}")
        else:
            print(f"Clustered nodes: {len(G)} (fully clustered)")
        print("Number of clusters: ", len(subgraphs))
        print("\n\n")
        
        def node2text(node_id, subgraph):
            data = subgraph[node_id]
            text = ""
            text += f"{data[ID_GTIN_COLUMN]:15}" + " "  # Adds the GTIN
            if show_node_id:
                text += f"{node_id:32}"
            text += "    "
            text += f"{data[TEXT_COLUMN]:80}"
            return text
            
        for i_iter, (label, subgraph) in enumerate(subgraphs.items()):
            if not_display_unknown and label == -1: 
                continue
            
            if (filter_node_ids is not None) and not any([node_id in filter_node_ids for node_id in subgraph.nodes()]):
                continue  # If filter_node_ids != [] and the subgraph contains none of them, skip
            
            gtin_true_counts = np.unique([subgraph[node_id][ID_GTIN_COLUMN] for node_id in subgraph.nodes()], return_counts=True)[1]
            most_representatives = subgraph.core(method=core_method, weight=weight)[:n_objs]
            
            if (n_objs == 1 or len(most_representatives) == 1) or (only_unknown and label != -1):
                # Show in one line
                node_id = most_representatives[0]
                print(f"label {label:3} [trueL/nn: {len(gtin_true_counts):3}/{len(subgraph):4}]:     " + node2text(node_id, subgraph))
            else:
                # Show in multiple lines
                print(f"label {label:3} [trueL/nn: {len(gtin_true_counts):3}/{len(subgraph):4}] ", '-'*20)
                for node_id in most_representatives:
                    print(' '*8, node2text(node_id, subgraph))
        
    def stream_dismember(self, Gs: list[CGraph]) -> CGraph:
        """
        Apply the cut_edges process across multiple graphs, 
        returning a graph with the intersection of their edges.
        The edges of the first graph will be used and trimmed by the others.

        Args:
            Gs (list[CGraph]): List of input graphs.
        
        Returns:
            CGraph: Graph containing the edge intersection of all graphs.
        """
        final_graph = Gs[0].copy()
        for graph in Gs[1:]:
            final_graph = CGraph.pattern(final_graph, graph, 'cut_edges')
        return final_graph
    
    def group_merging(self, graph: CGraph, merging_threshold:float, maximize:bool=True, iters:int=5, kneighbors:int=7, 
                      core_len:int=1, verbose:bool=False,model_name:str='sentence-transformers/all-MiniLM-L6-v2',
                      umap_kwargs={
                            'n_neighbors':5,
                            'min_dist':0.5,
                            'n_components':15,
                            'random_state':42,
                        }) -> CGraph:
        """
        Perform group merging based on similarity between representative components.

        Args:
            graph (CGraph): Graph with cluster labels to merge.
            merging_threshold (float): Edge similarity threshold for merging.
            maximize (bool, optional): 
                - If True, merge clusters when similarity > threshold.  
                - If False, merge clusters when similarity < threshold. Defaults to True.
            iters (int, optional): Number of iterations. Defaults to 5.
            kneighbors (int, optional): Number of nearest representative neighbors per iteration. Defaults to 7.
            core_len (int, optional): Number of representatives per cluster. Ensures the core is always merged. Defaults to 3.
            verbose (bool): prints output on terminal. By default it's False.

        Returns:
            CGraph: The graph with merged groups.
        """
        
        # Add core_len to kneighbors to ensure there are enough neighbors to cover the core
        kneighbors += core_len
        
        atual_graph = graph.copy()
        
        ## modelos utilizados
        model = SentenceTransformer(model_name)
        umap_model = umap.UMAP(metric='cosine',**umap_kwargs)
        
        for i_iter in range(iters):    
            # Filter to keep only representatives
            subgraphs = atual_graph.dismember()
            
            if verbose:
                print(f"iter {i_iter}")
                print(" "*4, f'dismembered into {len(subgraphs)} subgraphs')
            
            # Dictionary mapping each core tuple to its subgraph
            coretuple2subgraph = {
                tuple(subgraph.core(core_len)): subgraph for label, subgraph in subgraphs.items() if label != -1
            }
            
            # Collect only representative nodes
            only_representatives = atual_graph.subgraph(list(chain.from_iterable(coretuple2subgraph.keys())))
            if verbose:
                print("Number of only_representatives: ", len(only_representatives))
            
            ## calcula a similaridade entre os textos
            nodes_text_pairs = [(n, f"base: {d.get('produto_base')} detalhado: {d.get('produto_detalhado')}" ) for n, d in only_representatives.nodes(data=True)]
            texts = [t for n, t in nodes_text_pairs if t is not None]
            
            embeddings = model.encode(texts, convert_to_tensor=True)
            embeddings = normalize(embeddings,norm='l2') ## norm l2
            data_lowdim = umap_model.fit_transform(embeddings)
            
            ## euclidean distance matrix
            norms = np.sum(data_lowdim ** 2, axis=1, keepdims=True)
            dist_sq = norms + norms.T - 2 * np.dot(data_lowdim, data_lowdim.T)
            dist_sq = np.clip(dist_sq, a_min=0.0, a_max=None)
            dist_matrix = np.sqrt(dist_sq)
            
            sim_matrix = 1 / (1 + dist_matrix) ## similarity matrix
            # print("matriz de similaridade feita")
            ## cria as edges
            edges_to_add = []
            for i, j in itertools.combinations(range(len(nodes_text_pairs)), 2):
                n1 = nodes_text_pairs[i][0]
                n2 = nodes_text_pairs[j][0]
                score = float(sim_matrix[i, j])  # converter para float padrão Python (evita tensor)
                edges_to_add.append((n1, n2, {"similarity_score": score}))

            # Adiciona todas as arestas de uma vez (muito mais rápido)
            only_representatives = nx.Graph()
            only_representatives.add_nodes_from(
                (n, atual_graph.G.nodes[n]) for n in chain.from_iterable(coretuple2subgraph.keys())
            )
            only_representatives.add_edges_from(edges_to_add)
            only_representatives = CGraph(only_representatives)
            
            
            # Begin merging process
            same_groups = {}
            next_label = 0   # Unique label marker for groups
            for node_id in only_representatives.nodes():  # For each representative node 
                
                # If already regrouped, propagate label
                if node_id in same_groups.keys(): 
                    propag_label = same_groups[node_id]
                else:  # Otherwise, create a new label
                    propag_label = next_label
                    
                    # Assign the label to all nodes from the same core
                    for core_tuple in coretuple2subgraph.keys():
                        if node_id in core_tuple:
                            for core_id in core_tuple:
                                same_groups[core_id] = next_label
                    
                    next_label += 1
                
                # Get the k nearest neighbors
                nearests = only_representatives.direct_k_nearest_neighbors(node_id, k=kneighbors, weight='similarity_score')
                
                # Merging conditions:
                # maximize=True  -> score > threshold
                # maximize=False -> score < threshold
                for neighbor_id, score in nearests:
                    if not (maximize ^ (score > merging_threshold)):
                        same_groups[neighbor_id] = propag_label
            
            # Reorganize into {label: list of representative node_ids after merging}
            groups = {label: [] for label in same_groups.values()}
            for node_id, label in same_groups.items():
                groups[label].append(node_id)
            
            # Keep only groups with more than 1 representative
            groups = {label: node_ids for label, node_ids in groups.items() if len(node_ids) > 1}
            
            if verbose:
                print("Junção de grupos antigos: ")
                print(groups)
                for newlabel,groupas in groups.items():
                    print(f"newlabel: {newlabel} -------- ")
                    for old_node_id_representative in groupas:
                        d = only_representatives[old_node_id_representative]
                        print('  ',f"[{d['label']}] {d['original']:30} | {d['produto_base']:30} | {d['produto_detalhado']}")
            
            if len(groups) == 0:
                if verbose:
                    print("No mergings were performed")
                return atual_graph
            
            # Map each merging group to subgraphs
            mergelabel2subgraph = {}
            for subgraph_coretuple, subgraph in coretuple2subgraph.items():
                for new_label, merged_node_ids in groups.items():
                    if any([bool(mni in subgraph_coretuple) for mni in merged_node_ids]):  # If they match
                        if new_label in mergelabel2subgraph.keys():
                            mergelabel2subgraph[new_label].append(subgraph)
                        else:
                            mergelabel2subgraph[new_label] = [subgraph]
                        break  # Each subgraph belongs to only one group
            
            # Recalculate and reassign labels
            for new_label, subgraphs in mergelabel2subgraph.items():
                max_label = -2
                merged_node_ids = []
                for s in subgraphs:
                    max_label = max(max_label, int((next(iter(s.nodes(data='label'))))[1]))
                    merged_node_ids.extend(s.nodes())
                
                # Set all merged nodes to max_label
                node_setter = {node_id: max_label for node_id in merged_node_ids}
                nx.set_node_attributes(atual_graph.G, node_setter, 'label')  # Apply new labels
            
            # Display current representatives
            if verbose:
                print("\n"*15)
                pass
                # print("\n\n\n\nProcess finished, showing representatives")
                # self.show_representatives(atual_graph, n_objs=15,ID_GTIN_COLUMN='gtin')
            
        return atual_graph

CLUSTERING_DEFAULTS = SimpleNamespace(
    STRINGMATCH_THRESHOLD=0.9,                      ## threshold do stringmatch
    BERT_THRESHOLD=0.7,                             ## threshold do bert
    GRAPHBUILDER_FIELD_IDNFE="id_nfe",              ## coluna do dataframe utilizado para o id nfe
    GRAPHBUILDER_FIELD_STR="produto_detalhado",     ## coluna do dataframe utilizado como fonte do texto
    MAX_ITER_CLIQUE=30,                             ## quantidade de vezes que ele vai rodar o clique
    STEP_CLIQUE=3,                                  ## step de cada iteração do clique
    GROUP_MERGING_THRESHOLD=0.7,                    ## threshold mínimo para se fazer o merging de grupos
    MERGING_MAXIMIZE=True,                          ## a ideia é a fazer uma maximização da similaridade de merging
    MERGING_ITERS=1,                                ## quantidade de iterações que se faz o group merging
    MERGING_KN=5,                                   ## quantidade de grupos de representativos comparados por vez
    MERGING_CORE=1                                  ## quantidade de representativos por grupo
)

class NFeCluster():
    def __init__(self,
                 embedding_model:Union[str|SentenceTransformer]=None,
                 data:pd.DataFrame = None,
                 ):
        """
        Constructor of NFeCluster class.

        Args:
            embedding_model (str|SentenceTransformer): specifies the SentenceTransformer model used to embed texts. 
                By default, it's the string 'sentence-transformers/all-MiniLM-L6-v2', but if you 
                want to use a pre-installed model, you can pass a SentenceTransformer instance. See 
                more about the usage in the class similarity_graph.BERTEmbeddingEdgeGenerator with the 
                parameter 'model'. If it's None, the model won't be downloaded.
            
            data (list[str] | pd.DataFrame): specifies the data that will be clustered. By default, it's None, indicating
                that it will be passed after initialization using the method 'add_data' (it's recommended to use this 
                approach if you intend to add CGraphs to the data). Otherwise, you can pass a pandas DataFrame that must 
                follow the format required by similarity_graph.EdgeGenerator. In this case, a StringMatchEdgeGenerator will 
                be created using values from CLUSTERING_DEFAULTS.
        """
                
        self.embedding_model = embedding_model # or 'sentence-transformers/all-MiniLM-L6-v2'
        
        if self.embedding_model is not None:
            if type(self.embedding_model) == str:
                self.embedding_model = SentenceTransformer(self.embedding_model) ## downloads model
        
        ## dict in the format: {'dismember':[cgraph1,cgraph2...], 'merging':[cgraph1,cgraph2...]}
        ## graphs under 'dismember' are used to separate different types of clusters using their edges
        ## graphs under 'merging' are used to merge different clusters based on connections between cluster representatives. 
        ## See more in Analyser.group_merging
                
        self.graphs = {'dismember':[],'merging':[]} 
        
        if data is not None:
            ## Creates a StringMatchGenerator. Otherwise, assumes that data will be passed later
            if isinstance(data,pd.DataFrame):
                edge_gen = StringMatchEdgeGenerator(data)
                edges_df = edge_gen.generate_edges(field_str=CLUSTERING_DEFAULTS.GRAPHBUILDER_FIELD_STR)
                sm_graph = build_graph(data,edges_df,column_name=CLUSTERING_DEFAULTS.GRAPHBUILDER_FIELD_IDNFE)
                self.add_data(sm_graph,graph_type='dismember',not_nfe=False)
            else:
                ## Creates a DataFrame from data=list[str], considering that nfes are not passed
                nfes_fake = list(np.arange(len(data)))
                datadf = pd.DataFrame(data={CLUSTERING_DEFAULTS.GRAPHBUILDER_FIELD_IDNFE:nfes_fake, 
                                            CLUSTERING_DEFAULTS.GRAPHBUILDER_FIELD_STR:data})
                edge_gen = StringMatchEdgeGenerator(datadf)
                edges_df = edge_gen.generate_edges(field_str=CLUSTERING_DEFAULTS.GRAPHBUILDER_FIELD_STR)
                sm_graph = build_graph(datadf,edges_df,column_name=CLUSTERING_DEFAULTS.GRAPHBUILDER_FIELD_IDNFE)
                self.add_data(sm_graph,graph_type='dismember',not_nfe=True)
                
    
    def add_data(self,data:nx.Graph,graph_type:str,name:str=None,not_nfe:bool=False) -> None:
        """
        Add a graph built by one of the classes extended from similarity_graph.EdgeGenerator.
                
        Args:
            data (networkx.Graph): graph to be added.
            graph_type (str): the type of the graph ('dismember' or 'merging').
            name (str): name of the graph. Default is None.
            not_nfe(bool): if true, considers that you will not pass nfe. By default it's False.
        """
                
        
        if (not (graph_type=='dismember' or graph_type=='merging')):
            raise ValueError(f"graph_type '{graph_type}' is not allowed. It must be 'dismember' or 'merging'")
        
        self.nfes_provided = not not_nfe
        self.graphs[graph_type].append(CGraph(data,name=name))

    def cluster(self,apply_merging=False, clustering_method='deep_clique',clustering_method_params={}):
        """
        Perform graph clustering using the specified method and optionally apply a merging phase.
        
        This function handles clustering of a dismembered graph using different strategies.
        By default, it uses the "deep_clique" method unless another clustering algorithm 
        is specified. After clustering, an optional merging phase can be applied to 
        refine and combine clusters.
        
        Args:
            apply_merging (bool, optional): Whether to apply the merging phase after clustering. 
                Defaults to False.
            clustering_method (str, optional): The clustering algorithm to use. 
                Defaults to 'deep_clique'.
            clustering_method_params (dict, optional): Parameters for the clustering method.
                Defaults to an empty dict.

        Returns:
            tuple:
                - dict: A mapping from node ID (`id_nfe`) to its assigned cluster label.
                - networkx.Graph: The clustered graph with labels assigned to nodes.
        """
        analyser = Analyser()
        if len(self.graphs['dismember']) > 1:
            all_cutted = analyser.stream_dismember(self.graphs['dismember'])
        else:
            all_cutted = self.graphs['dismember'][0]
        
        ## apply clustering
        if clustering_method == 'deep_clique':
            if len(clustering_method_params) == 0:
                analyser.deep_clique(
                    all_cutted,
                    max_iter=CLUSTERING_DEFAULTS.MAX_ITER_CLIQUE,
                    step=CLUSTERING_DEFAULTS.STEP_CLIQUE
                )
            else:
                analyser.deep_clique(all_cutted, **clustering_method_params)
        else:
            ## assumes that the clustering method already contains the rest of the implementation
            all_cutted.nxcluster(clustering_method, **clustering_method_params)
        
        ## apply merging if requested
        # print("\n\n\n\Antes do merging:")
        # analyser.show_representatives(all_cutted, n_objs=5,ID_GTIN_COLUMN='gtin')
        # print("\n"*15)
        if apply_merging:
            # if len(self.graphs['merging']) == 0:
            #     print("There are no graphs of 'merging' type. Ignoring merging phase")
            #     pass
            # else:
            try:
                all_cutted = analyser.group_merging(
                    all_cutted,
                    verbose=True,
                    merging_threshold=CLUSTERING_DEFAULTS.GROUP_MERGING_THRESHOLD,
                    maximize=CLUSTERING_DEFAULTS.MERGING_MAXIMIZE,
                    iters=CLUSTERING_DEFAULTS.MERGING_ITERS,
                    kneighbors=CLUSTERING_DEFAULTS.MERGING_KN,
                    core_len=CLUSTERING_DEFAULTS.MERGING_CORE
                )
            except Exception as e:
                pass
        ## prepare output
        ## the output is in the format: id_nfe:label
        idnfe2label = dict(all_cutted.nodes(data='label'))  ## note: node ID is the same as id_nfe
        
        ## If nfes were not provided (they are fake), the output will be sorted based on field nfe
        # print("not self.nfes_provided: ",not self.nfes_provided)
        if not self.nfes_provided:
            sorted_idx = list(sorted(list(idnfe2label.keys())))
            unsorted_labels = list(idnfe2label.values())
            only_labels = [str(unsorted_labels[idx]) for idx in sorted_idx]
            return only_labels, all_cutted ## list[str] , CGraph
        return idnfe2label, all_cutted

    @classmethod
    def pipeline(cls,data:pd.DataFrame,merge=False,
                 price_diff=3.0):
        """
        Applies a pipeline in dataframe, considerando as colunas:
        - id_item: tratado como o id
        - preco: utilizado para separar por value difference (usa o price_diff)
        - produto_base_clear: para separar com stringmatch
        - produto_detalhado: caso merge=True, realiza o merge utilizando esse campo
        """
        ID_COLUMN = 'id_item'
        PRECO = 'preco'
        BASE_CLEAR = 'produto_base_clear'
        DETALHADO = 'produto_detalhado'
        
        ## tratamento de None/NaN:
        values = {
            PRECO:-2*price_diff, ## considera como sendo um preço inválido
            BASE_CLEAR:'',
            DETALHADO:'',
        }
        data = data.fillna(values)
        clusterer = cls(data)
        
        # ## preco
        # edge_gen = ValueDifferenceEdgeGenerator(data)
        # edges_df = edge_gen.generate_edges(price_diff,field_valor=PRECO,ID_NFE_COLUMN=ID_COLUMN)
        # price_graph = build_graph(data,edges_df,column_name=ID_COLUMN)
        # clusterer.add_data(price_graph,graph_type='dismember',not_nfe=False)
        
        ## produto_base_clear
        edge_gen = StringMatchEdgeGenerator(data)
        edges_df = edge_gen.generate_edges(field_str=BASE_CLEAR,ID_NFE_COLUMN=ID_COLUMN,threshold=0.9)
        pbc_graph = build_graph(data,edges_df,column_name=ID_COLUMN)
        clusterer.add_data(pbc_graph,graph_type='dismember',name='stringmatch',not_nfe=False)
        
        ## produto_base_detalhado
        
        
        ## Aplica o método de clusterização
        id2label, output_graph = clusterer.cluster(apply_merging=merge,
                                                   clustering_method='label_propagation')
        
        
        aaaa = output_graph.dismember()
        data['clusters'] = data[ID_COLUMN].map(lambda id__: id2label[id__])
        return data