#!/usr/bin/python3
# By Steve Hanov, 2011. Released to the public domain.
# Please see http://stevehanov.ca/blog/index.php?id=115 for the accompanying article.
#
# Based on Daciuk, Jan, et al. "Incremental construction of minimal acyclic finite-state automata."
# Computational linguistics 26.1 (2000): 3-16.
#
# Updated 2014 to use DAWG as a mapping; see
# Kowaltowski, T.; CL. Lucchesi (1993), "Applications of finite automata representing large vocabularies",
# Software-Practice and Experience 1993

# https://gist.github.com/smhanov/94230b422c2100ae4218
# http://stevehanov.ca/blog/index.php?id=115

import sys
import time

# This class represents a node in the directed acyclic word graph (DAWG). It
# has a list of edges to other nodes. It has functions for testing whether it
# is equivalent to another node. Nodes are equivalent if they have identical
# edges, and each identical edge leads to identical states. The __hash__ and
# __eq__ functions allow it to be used as a key in a python dictionary.

class DawgNode:

    next_id = 0

    def __init__(self):
        self.id = DawgNode.next_id
        DawgNode.next_id += 1
        self.final = False
        self.edges = {}

        # Number of end nodes reachable from this one.
        self.count = 0

    def __str__(self):
        arr = []
        if self.final:
            arr.append("1")
        else:
            arr.append("0")

        for (label, node) in self.edges.items():
            arr.append(label)
            arr.append(str(node.id))

        return "_".join(arr)

    def __hash__(self):
        return self.__str__().__hash__()

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def num_reachable(self):

        # if a count is already assigned, return it
        if self.count: return self.count

        # count the number of final nodes that are reachable from this one.
        # including self
        count = 0
        if self.final: count += 1
        for label, node in self.edges.items():
            count += node.num_reachable() # recursively thru reachable nodes
        self.count = count
        return count

class Dawg:

    def __init__(self):

        self.previous_word = ""
        self.root = DawgNode()

        # Here is a list of nodes that have not been checked for duplication.
        self.unchecked_nodes = []

        # Here is a list of unique nodes that have been checked for
        # duplication.
        self.minimized_nodes = {}

        # Here is the data associated with all the nodes
        self.data = []

    def insert(self, word, data):

        assert word >= self.previous_word, "Error: Words must be inserted in alphabetical order."

        # find common prefix between word and previous word
        common_prefix = 0
        # loop over shorter word
        for i in range(min(len(word), len(self.previous_word))):
            if word[i] != self.previous_word[i]: break
            common_prefix += 1

        # Check the unchecked_nodes for redundant nodes, proceeding from last
        # one down to the common prefix size. Then truncate the list at that
        # point.
        self._minimize(common_prefix)
        self.data.append(data)

        # add the suffix, starting from the correct node mid-way through the
        # graph
        if len(self.unchecked_nodes) == 0:
            node = self.root
        else:
            node = self.unchecked_nodes[-1][2]

        # create new node, add to edges & to list of unchecked nodes
        # move to new node and repeat until end of word
        for letter in word[common_prefix:]:
            next_node = DawgNode()
            node.edges[letter] = next_node
            self.unchecked_nodes.append((node, letter, next_node))
            node = next_node

        node.final = True
        self.previous_word = word

    # minimize all unchecked_nodes, then
    # go through entire structure and assign the counts to each node.
    def finish(self):
        self._minimize(0);
        self.root.num_reachable()

    def _minimize(self, down_to):
        # proceed from the leaf up to a certain point
        for i in range(len(self.unchecked_nodes) - 1, down_to - 1, -1):
            (parent, letter, child) = self.unchecked_nodes[i];
            if child in self.minimized_nodes:
                # replace the child with the previously encountered one
                parent.edges[letter] = self.minimized_nodes[child]
            else:
                # add the state to the minimized nodes.
                self.minimized_nodes[child] = child;
            self.unchecked_nodes.pop()

    def lookup(self, word):
        node = self.root
        skipped = 0 # keep track of number of final nodes that we skipped
        for letter in word:
            if letter not in node.edges: return None
            for label, child in sorted(node.edges.items()):
                if label == letter:
                    # seen a word ending, save
                    if node.final: skipped += 1
                    # move to next node
                    node = child
                    break # > move back to letter loop in word
                skipped += child.count

        if node.final:
            return self.data[skipped] 

    def node_count(self):
        return len(self.minimized_nodes)

    def edge_count(self):
        count = 0
        for node in self.minimized_nodes:
            count += len(node.edges)
        return count

    def display(self):
        stack = [self.root]
        done = set()
        while stack:
            node = stack.pop()
            if node.id in done: continue
            done.add(node.id)
            print("{}: ({})".format(node.id, node))
            for label, child in node.edges.items():
                print("    {} goto {}".format(label, child.id))
                stack.append(child)

if __name__ == '__main__':

    DICTIONARY = "british-english.txt"
    # QUERY = sys.argv[1:]
    QUERY = ["things", "lechs", "eerie"]

    dawg = Dawg()
    WordCount = 0
    with open(DICTIONARY, "rt") as f:
        words = []
        for w in f.read().split():
            words.append(w.rstrip())
    words.sort()

    start = time.time()
    for word in words:
        WordCount += 1
        # insert all words, using the reversed version as the data associated with it
        dawg.insert(word, ''.join(reversed(word)))
        if (WordCount % 100) == 0:
            print("{0}\r".format(WordCount), end="")

    dawg.display()
    dawg.finish()
    print('-'*40)
    print("Dawg creation took {0} s".format(time.time()-start))

    edge_count = dawg.edge_count()
    print("Read {0} words into {1} nodes and {2} edges".format(
        WordCount, dawg.node_count(), edge_count))

    print("This could be stored in as little as {0} bytes".format(edge_count * 4))

    for word in QUERY:
        result = dawg.lookup(word)
        if result == None:
            print("{0}? Not in dictionary.".format(word))
        else:
            print("{0}? In the dictionary and has data {1}".format(word, result))
