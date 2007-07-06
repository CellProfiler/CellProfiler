#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "error.h"
#include "intset.h"
#include "bplus.h"

#if 0
#define TRACE
#endif

/* Allocate a new node and return its ID.  The node is filled with zeros. */
static int alloc_node(struct bplus *tree)
{
     return intset_pop(&tree->free_nodes);
}

static void free_node(struct bplus *tree, int i)
{
     assert(i > 0);
     intset_add(&tree->free_nodes, i);
     memset(&tree->nodes[i], 0, sizeof(union node));
}

void
init_bplus(struct bplus *tree, int max_leaves, int key_offset)
{
     struct internal *root;
     int max_nodes;

     assert(tree);
     max_nodes = 2 * max_leaves;
     tree->nodes = malloc(max_nodes * sizeof(union node));
     if (!tree->nodes)
	  fatal_perror("malloc");
     memset(tree->nodes, 0, max_nodes * sizeof(union node));
     init_intset(&tree->free_nodes, max_nodes);
     intset_fill(&tree->free_nodes);
     intset_remove(&tree->free_nodes, 0); /* Don't use 0 as a node ID. */
     tree->key_offset = key_offset;
     tree->rootid = alloc_node(tree);
     root = &tree->nodes[tree->rootid].internal;
     root->p[0] = -alloc_node(tree);
}

/**********************************************************************/

__inline__ static int node_key(const struct bplus *tree, int nodeid)
{
     void *value;
     struct leaf *leaf;

     assert(tree);
     assert(nodeid != 0);
     if (nodeid < 0) {
          leaf = &tree->nodes[-nodeid].leaf;
          value = leaf->v[0];
          return value ? bplus_tree_key(tree, value) : 
               (leaf->next ? node_key(tree, leaf->next) : 0);
     }
     else
          return tree->nodes[nodeid].internal.k[0];
}

static int insert(struct bplus *tree, int nodeid, void *value);

/* If there is room, insert the new value somewhere below this node
 * and return 0.  If not, split this node, insert value into the new
 * sibling, and return the new sibling's ID. */
static int insert_internal(struct bplus *tree, int internalid, void *value)
{
     struct internal *internal, *newsibling;
     int newchildid, newsiblingid, i, j;
     int lastp, lastk;

     internal = &tree->nodes[internalid].internal;
     i = bplus_internal_search(tree, internalid, bplus_tree_key(tree, value));
     assert(i >= 0 && i < IVALUES);
     assert(internal->p[i]);
     newchildid = insert(tree, internal->p[i], value);
     /* If the new key was lower than the lowest key in the child, we
      * have to decrease internal->k[i]. */
     internal->k[i] = node_key(tree, internal->p[i]);

     if (newchildid) {
	  /* The child node was split; newchild is its new right
           * sibling.  Rearrange internal and insert its pointer and
           * key in the proper places. */
	  lastp = internal->p[IVALUES - 1];
	  lastk = internal->k[IVALUES - 1];
	  for (j = IVALUES - 1; j >= i + 2; j--) {
	       internal->p[j] = internal->p[j - 1];
	       internal->k[j] = internal->k[j - 1];
	  }
	  if (i < IVALUES - 1) {
	       internal->p[i + 1] = newchildid;
	       internal->k[i + 1] = node_key(tree, newchildid);
	  } else {
	       lastp = newchildid;
	       lastk = node_key(tree, newchildid);
	  }
	  if (lastp) {
	       /* Full; must split internal. */
	       newsiblingid = alloc_node(tree);
	       newsibling = &tree->nodes[newsiblingid].internal;
	       for (j = 0, i = (IVALUES + 1) / 2; i < IVALUES; i++) {
		    if (internal->p[i]) {
			 newsibling->p[j] = internal->p[i];
			 newsibling->k[j] = internal->k[i];
			 internal->p[i] = 0;
			 internal->k[i] = 0; /* deadbeef? */
			 j++;
		    }
	       }
	       newsibling->p[j] = lastp;
	       newsibling->k[j] = lastk;
	       return newsiblingid;
	  }
     }
     return 0;
}

/* Insert a value.  If the node is full, split it and return the ID of
 * the new sibling; otherwise, return 0. */
static int insert_leaf(struct bplus *tree, int leafid, void *value)
{
     struct leaf *leaf;
     int siblingid, new_key;
     struct leaf *sibling;
     int i, j;
     void *tmp;
     int next, full;

     assert(leafid < 0);
     leaf = &tree->nodes[-leafid].leaf;
     full = leaf->v[LVALUES - 1] != 0;
     
     next = leaf->next;
     leaf->next = 0;
     new_key = bplus_tree_key(tree, value);
     for (i = 0; i < LVALUES + 1; i++) {
	  if (leaf->v[i] == 0 || bplus_tree_key(tree, leaf->v[i]) > new_key) {
	       tmp = leaf->v[i];
	       leaf->v[i] = value;
	       value = tmp;
	       break;
	  }
     }
     for (i++; i < LVALUES + 1; i++) {
	  tmp = leaf->v[i];
	  leaf->v[i] = value;
	  value = tmp;
     }

     if (!full) {
	  leaf->next = next;
	  return 0;
     }

     /* Full; must split. */
     siblingid = -alloc_node(tree);
     sibling = &tree->nodes[-siblingid].leaf;
     
     for (i = (LVALUES + 1) / 2, j = 0; i < LVALUES + 1; i++, j++) {
	  sibling->v[j] = leaf->v[i];
	  leaf->v[i] = 0;
     }

     sibling->next = next;
     leaf->next = siblingid;
     return siblingid;
}

static int insert(struct bplus *tree, int nodeid, void *value)
{
     assert(value != NULL);
     if (nodeid < 0)
	  return insert_leaf(tree, nodeid, value);
     else if (nodeid > 0)
	  return insert_internal(tree, nodeid, value);
     else
	  abort();
}

void bplus_insert(struct bplus *tree, void *value)
{
     int newnodeid, newrootid;
     struct internal *root, *newroot;

     assert(tree);
     assert(value);
#ifdef TRACE
     printf("(trace) bplus_insert %d(%p)\n", bplus_tree_key(tree, value), 
            value);
#endif
     newnodeid = insert_internal(tree, tree->rootid, value);
     if (newnodeid) {
	  /* Need to split root. */
          root = &tree->nodes[tree->rootid].internal;
	  newrootid = alloc_node(tree);

	  newroot = &tree->nodes[newrootid].internal;
          newroot->k[0] = root->k[0];
	  newroot->p[0] = tree->rootid;

	  newroot->k[1] = node_key(tree, newnodeid);
	  newroot->p[1] = newnodeid;

	  tree->rootid = newrootid;
     }
#ifdef TRACE
     bplus_print_tree(tree);
#endif
}

/**********************************************************************/

/* Returns -1 if value not found; 0 if value was deleted, but leaf is
 * not underfull; and 1 if value was deleted and leaf is underfull. */
__inline__ static int delete_leaf(struct bplus *tree, int nodeid, void *value)
{
     int i, fill;
     struct leaf *leaf;

     assert(tree);
     assert(nodeid < 0);
     leaf = &tree->nodes[-nodeid].leaf;
     
     /* Delete the value if it is here. */
     for (i = 0; i < LVALUES; i++)
	  if (leaf->v[i] == value)
	       break;
     if (i == LVALUES)
	  return -1; /* Not here. */
     for (; i < LVALUES - 1; i++)
	  leaf->v[i] = leaf->v[i + 1];
     leaf->v[LVALUES - 1] = NULL;

     /* Count the number of values and return true if leaf is underfull. */
     fill = 0;
     for (i = 0; i < LVALUES; i++)
	  if (leaf->v[i])
	       fill++;
     return fill < LVALUES / 2.0;
}

/* Returns true iff internal node is underfull */
static int internal_underfull_p(const struct internal *internal)
{
     int i;
     assert(internal);
     for (i = 0; i < IVALUES && internal->p[i]; i++);
     return (i < IVALUES / 2.0);
}

/* Merge two leaf nodes into one (the first one).  Return true iff the
 * second node has been freed. */
static int merge_leaf(struct bplus *tree, int leftid, int rightid)
{
     struct leaf *left, *right;
     void *v[2 * LVALUES];
     int values, i, j;

     assert(tree);
     assert(leftid < 0 && rightid < 0);
     left = &tree->nodes[-leftid].leaf;
     right = &tree->nodes[-rightid].leaf;

     values = 0;
     for (i = 0; i < LVALUES && left->v[i]; i++)
	  v[values++] = left->v[i];
     for (i = 0; i < LVALUES && right->v[i]; i++)
	  v[values++] = right->v[i];

     if (values <= LVALUES) {
	  for (i = 0; i < values; i++)
	       left->v[i] = v[i];
	  for (; i < LVALUES; i++)
	       left->v[i] = NULL;
	  left->next = right->next;
	  free_node(tree, -rightid);
	  return 1;
     } else {
	  for (i = 0, j = 0; i < LVALUES / 2.0; i++, j++)
	       left->v[i] = v[j];
	  for (; i < LVALUES; i++)
	       left->v[i] = NULL;
	  for (i = 0; j < values; i++, j++)
	       right->v[i] = v[j];
	  for (; i < LVALUES; i++)
	       right->v[i] = NULL;
	  return 0;
     }
}

static int merge_internal(struct bplus *tree, int leftid, int rightid)
{
     struct internal *left, *right;
     int p[2 * IVALUES];
     int k[2 * IVALUES];
     int values, i, j;

     assert(tree);
     assert(leftid > 0 && rightid > 0);
     left = &tree->nodes[leftid].internal;
     right = &tree->nodes[rightid].internal;

     values = 0;
     for (i = 0; i < IVALUES && left->p[i]; i++, values++) {
	  p[values] = left->p[i];
	  k[values] = left->k[i];
     }
     for (i = 0; i < IVALUES && right->p[i]; i++, values++) {
	  p[values] = right->p[i];
	  k[values] = right->k[i];
     }

     if (values <= IVALUES) {
	  for (i = 0; i < values; i++) {
	       left->p[i] = p[i];
	       left->k[i] = k[i];
	  }
	  for (; i < IVALUES; i++) {
	       left->p[i] = 0;
	       left->k[i] = 0; /* deadbeef? */
	  }
	  free_node(tree, rightid);
	  return 1;
     } else {
	  for (i = 0, j = 0; i < IVALUES / 2; i++, j++) {
	       left->p[i] = p[j];
	       left->k[i] = k[j];
	  }
	  for (; i < IVALUES; i++) {
	       left->p[i] = 0;
	       left->k[i] = 0; /* deadbeef? */
	  }
	  for (i = 0; j < values; i++, j++) {
	       right->p[i] = p[j];
	       right->k[i] = k[j];
	  }
	  for (; i < IVALUES; i++) {
	       right->p[i] = 0;
	       right->k[i] = 0; /* deadbeef? */
	  }
	  return 0;
     }
}

/* Returns true iff the right child was removed */
static int merge(struct bplus *tree, int leftid, int rightid)
{
     if (leftid > 0 && rightid > 0)
	  return merge_internal(tree, leftid, rightid);
     else if (leftid < 0 && rightid < 0)
	  return merge_leaf(tree, leftid, rightid);
     else
	  abort();
}

/* Merge two nodes and update the keys in their parent. */
static void merge_and_fixup(struct bplus *tree, int internalid, 
			    int pointer_index)
{
     int childid, nextchildid, i;
     struct internal *internal;

     assert(tree);
     assert(internalid > 0);
     assert(0 <= pointer_index && pointer_index < IVALUES - 1);
     internal = &tree->nodes[internalid].internal;
     childid = internal->p[pointer_index];
     nextchildid = internal->p[pointer_index + 1];
     if (!nextchildid) {
	  internal->k[pointer_index] = node_key(tree, childid);
	  return;
     }
     if (merge(tree, childid, nextchildid)) {
          /* Nextchild was freed. */
	  for (i = pointer_index + 1; i < IVALUES - 1; i++) {
	       internal->p[i] = internal->p[i + 1];
	       internal->k[i] = internal->k[i + 1];
	  }
	  internal->p[IVALUES - 1] = 0;
	  internal->k[IVALUES - 1] = 0; /* deadbeef? */
     } else {
          /* Nextchild is still there. */
          internal->k[pointer_index + 1] = node_key(tree, nextchildid);
     }
     internal->k[pointer_index] = node_key(tree, childid);
}

static void internal_fixup(struct bplus *tree, int internalid, int i)
{
     struct internal *internal;
     int childid;

     assert(tree);
     assert(internalid > 0);
     assert(i >= 0 && i < IVALUES);
     internal = &tree->nodes[internalid].internal;
     childid = internal->p[i];
     assert(childid != 0);
     internal->k[i] = node_key(tree, childid);
}

__inline__ static int delete(struct bplus *tree, int nodeid, void *value);
     
/* Returns -1 if value not found; 0 if value was deleted, but leaf is
 * not underfull; and 1 if value was deleted and leaf is underfull. */
__inline__ static int delete_internal(struct bplus *tree, int nodeid, 
                                      void *value)
{
     int key;
     struct internal *internal;
     int i; /* Position of the child with the key. */

     assert(nodeid > 0);
     internal = &tree->nodes[nodeid].internal;

     key = bplus_tree_key(tree, value);
     i = bplus_internal_search(tree, nodeid, key);
     if (i == 0 && key < internal->k[0])
          return -1; /* Not here, so we're done */

     if (i > 0)
          i--;

     /* The following is a mess...  Its purpose it to ensure that
      * children that might contain values matching the key are
      * visited in turn, so that the object (value) that is to be
      * deleted can be found. */
     for (;;) {
	  if (!internal->p[i] || internal->k[i] > key)
	       return -1; /* We're done */
	  switch (delete(tree, internal->p[i], value)) {
	  case 0: 
               internal_fixup(tree, nodeid, i);
               return 0; /* Child is not underfull, so we're done */
	  case 1: 
               internal_fixup(tree, nodeid, i);
               goto underfull;
	  case -1: 
               /* The child did not contain the value. */
	       if (i == IVALUES - 1)
		    return -1;
	       else
		    i++;
	  }
     }
 underfull:
     /* Child is underfull; merge with sibling */
     if (i + 1 < IVALUES && internal->p[i + 1]) {
	  /* Child is not the last child. */
	  merge_and_fixup(tree, nodeid, i);
     } else if (i > 0)
	  merge_and_fixup(tree, nodeid, i - 1);
     return internal_underfull_p(internal);
}

__inline__ static int delete(struct bplus *tree, int nodeid, void *value)
{

     assert(value != NULL);
     if (nodeid < 0)
	  return delete_leaf(tree, nodeid, value);
     else if (nodeid > 0)
	  return delete_internal(tree, nodeid, value);
     else
	  abort();
}

void bplus_delete(struct bplus *tree, void *value)
{
     struct internal *root;
     int newrootid;

     assert(value);
#ifdef TRACE
     printf("(trace) bplus_delete %d(%p)\n", bplus_tree_key(tree, value), 
            value);
#endif
     delete_internal(tree, tree->rootid, value);
     /* If the root has only one child, and that child is an internal
      * node, remote the root and make its child the new root. */
     root = &tree->nodes[tree->rootid].internal;
     if (root->p[1] == 0 && root->p[0] > 0) {
	  newrootid = root->p[0];
	  free_node(tree, tree->rootid);
	  tree->rootid = newrootid;
     }
}

/**********************************************************************/

int bplus_member(const struct bplus *tree, void *value)
{
     int nodeid, i, key;
     struct internal *internal;
     struct leaf *leaf;

     nodeid = tree->rootid;
     key = bplus_tree_key(tree, value);
     while (nodeid > 0) {
	  internal = &tree->nodes[nodeid].internal;
	  for (i = -1; 
	       i < IVALUES - 1 && internal->p[i + 1] &&
		    internal->k[i + 1] <= key;
	       i++);
	  if (i == -1)
	       return 0;
	  nodeid = internal->p[i];
     }
     assert(nodeid < 0);
     leaf = &tree->nodes[-nodeid].leaf;
     for (i = 0; i < LVALUES; i++) {
	  if (!leaf->v[i])
	       return 0;
	  if (leaf->v[i] == value)
	       return 1;
     }
     return 0;
}


void *bplus_search_fake(const struct bplus *tree_arg, int key)
{
     struct internal *internal;
     static const struct bplus *tree;
     static struct leaf *leaf;
     static int i;

     if (tree_arg) {
	  tree = tree_arg;
	  internal = &tree->nodes[tree->rootid].internal;
	  while (internal->p[0] > 0)
	       internal = &tree->nodes[internal->p[0]].internal;
	  assert(internal->p[0] < 0);
	  leaf = &tree->nodes[-internal->p[0]].leaf;
	  i = -1;
     }

     for (;;) {
	  i++;
	  if (i == LVALUES || !leaf->v[i]) {
	       if (!leaf->next)
		    return NULL;
	       leaf = &tree->nodes[-leaf->next].leaf;
	       i = 0;
	  }
	  if (leaf->v[i] && bplus_tree_key(tree, leaf->v[i]) >= key)
	       break;
     }
     return leaf->v[i];
}

static void print_tree_leaf(const struct bplus *tree, int nodeid)
{
     int i;
     struct leaf *leaf;

     assert(tree);
     assert(nodeid < 0);
     leaf = &tree->nodes[-nodeid].leaf;
     for (i = 0; i < LVALUES; i++) {
          if (!leaf->v[i])
               break;
          printf("%d ", bplus_tree_key(tree, leaf->v[i]));
     }
     printf("-- ");
     for (i = 0; i < LVALUES; i++) {
          if (!leaf->v[i])
               break;
          printf("%p ", leaf->v[i]);
     }
     putchar('\n');
}

static void print_tree_1(const struct bplus *tree, int nodeid, int level);

static void print_tree_internal(const struct bplus *tree, int nodeid, int level)
{
     int i;
     struct internal *internal;

     assert(tree);
     assert(nodeid > 0);
     assert(level >= 0);
     internal = &tree->nodes[nodeid].internal;
     for (i = 0; i < IVALUES; i++) {
          if (!internal->p[i])
               break;
          printf("%d(%d) ", internal->k[i], internal->p[i]);
     }
     putchar('\n');
     for (i = 0; i < IVALUES; i++) {
          if (!internal->p[i])
               break;
          print_tree_1(tree, internal->p[i], level + 1);
     }
}

static void print_tree_1(const struct bplus *tree, int nodeid, int level)
{
     int i;

     for (i = 0; i < level; i++)
          printf("   ");
     printf("%6d: ", nodeid);
     if (nodeid > 0)
          print_tree_internal(tree, nodeid, level);
     else
          print_tree_leaf(tree, nodeid);
}

void bplus_print_tree(const struct bplus *tree)
{
     print_tree_1(tree, tree->rootid, 0);
}

void bplus_print_all_values(const struct bplus *tree)
{
     struct internal *internal;
     struct leaf *leaf;
     int i;

     assert(tree);
     internal = &tree->nodes[tree->rootid].internal;
     while (internal->p[0] > 0)
          internal = &tree->nodes[internal->p[0]].internal;
     leaf = &tree->nodes[-internal->p[0]].leaf;
 again:
     for (i = 0; i < LVALUES; i++)
          if (leaf->v[i])
               printf("%d ", bplus_tree_key(tree, leaf->v[i]));
     printf("\n");
     if (leaf->next) {
          leaf = &tree->nodes[-leaf->next].leaf;
          goto again;
     }
}

/**********************************************************************/

__inline__ static int bplus_size_1(const struct bplus *tree, int nodeid)
{
     int size, i;
     struct internal *internal;
     struct leaf *leaf;

     size = 0;
     if (nodeid > 0) {
	  internal = &tree->nodes[nodeid].internal;
	  for (i = 0; i < IVALUES; i++)
	       if (internal->p[i] || i == 0)
		    size += bplus_size_1(tree, internal->p[i]);
     } else {
	  assert(nodeid < 0);
	  leaf = &tree->nodes[-nodeid].leaf;
	  for (i = 0; i < LVALUES; i++)
	       if (leaf->v[i])
		    size++;
     }
     return size;
}

int bplus_size(const struct bplus *tree)
{
     return bplus_size_1(tree, tree->rootid);
}
