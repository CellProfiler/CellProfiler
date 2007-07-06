#ifndef VLCUTILS_BPLUS_H
#define VLCUTILS_BPLUS_H

#include "intset.h"

struct bplus {
     union node *nodes;
     struct intset free_nodes;
     int rootid;
     int key_offset;
};

/* Branching factor (number of pointers and values in internal nodes). */
#define IVALUES 9
/* Number of values in each leaf node. */
#define LVALUES 17

struct internal {
     int k[IVALUES];
     int p[IVALUES];
};

struct leaf {
     void *v[LVALUES];
     int next;
};

union node {
     struct internal internal;
     struct leaf leaf;
};

void
init_bplus(struct bplus *tree, int max_leaves, int key_offset);

void bplus_insert(struct bplus *tree, void *value);

/* Delete a specific object */
void bplus_delete(struct bplus *tree, void *value);

/* Return true iff the value exists in the tree. */
int bplus_member(const struct bplus *tree, void *value);

/* Returns the number of values in the tree. */
int bplus_size(const struct bplus *tree);

void bplus_print_tree(const struct bplus *tree);

#define bplus_tree_key(tree, value) (*(int *)((char *)value + tree->key_offset))

/* Returns an index i indicating that the new value should be inserted
 * into the node with ID internal->p[i]. */
__inline__ static int bplus_internal_search(const struct bplus *tree, int internalid, int key)
{
     struct internal *internal;

     assert(tree);
     assert(internalid > 0);
     internal = &tree->nodes[internalid].internal;

     /* FIXME: Replace this with something more general. */
     assert(IVALUES == 9);
     if (!internal->p[1] || internal->k[1] >= key)
          return 0;
     if (!internal->p[2] || internal->k[2] >= key)
          return 1;
     if (!internal->p[3] || internal->k[3] >= key)
          return 2;
     if (!internal->p[4] || internal->k[4] >= key)
          return 3;
     if (!internal->p[5] || internal->k[5] >= key)
          return 4;
     if (!internal->p[6] || internal->k[6] >= key)
          return 5;
     if (!internal->p[7] || internal->k[7] >= key)
          return 6;
     if (!internal->p[8] || internal->k[8] >= key)
          return 7;
#ifndef NDEBUG
     if (internal->p[8])
#endif
          return 8;
     abort();
}

/* Return the first value in the tree that has a key greater or equal
   to KEY.  (Can subsequently be called with TREE_ARG equal to NULL to
   get the dollowing values.  However, the results are undefined if
   the tree is modified between calls.) */
__inline__ static
void *bplus_search(const struct bplus *tree_arg, int key)
{
     struct internal *internal;
     int nodeid;
     static const struct bplus *tree;
     static int i;
     static struct leaf *leaf;

     if (tree_arg) {
	  tree = tree_arg;
	  leaf = NULL;
	  nodeid = tree->rootid;
	  while (nodeid > 0) {
	       internal = &tree->nodes[nodeid].internal;
               i = bplus_internal_search(tree, nodeid, key);
               nodeid = internal->p[i];
               i = -1;
	  }
          leaf = &tree->nodes[-nodeid].leaf;
     }
     if (!leaf)
          return NULL;
     for (;;) {
          i++;
          if (i == LVALUES) {
               if (!leaf->next)
                    return leaf = NULL;
               leaf = &tree->nodes[-(leaf->next)].leaf;
               i = 0;
          }
          if (leaf->v[i] && bplus_tree_key(tree, leaf->v[i]) >= key)
              return leaf->v[i];
     }
     
}

#endif
