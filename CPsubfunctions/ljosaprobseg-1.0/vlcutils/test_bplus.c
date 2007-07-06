#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "error.h"
#include "bplus.c"

struct object {
     int data;
     int key;
};

void *make_object(int data, int key)
{
     struct object *object;

     object = malloc(sizeof(struct object));
     if (!object)
          fatal_perror("malloc");
     object->data = data;
     object->key = key;
     return object;
}

struct bplus tree;

void check_leaf_linkage(void)
{
     int i;
     struct object *o;

     i = 0;
     o = bplus_search(&tree, 0);
     while (o) {
          i++;
          o = bplus_search(NULL, 0);
     }
     assert(i == bplus_size(&tree));
}

void one_node(void)
{
     struct object *o1, *o2;

     o1 = make_object(1, 1);
     assert(!bplus_member(&tree, o1));
     bplus_insert(&tree, o1);
     bplus_print_tree(&tree);
     assert(bplus_size(&tree) == 1);
     assert(bplus_member(&tree, o1));

     o2 = make_object(2, 1);
     assert(!bplus_member(&tree, o2));
     bplus_insert(&tree, o2);
     bplus_print_tree(&tree);
     assert(bplus_size(&tree) == 2);
     assert(bplus_member(&tree, o1));
     assert(bplus_member(&tree, o2));

     bplus_delete(&tree, o1);
     bplus_print_tree(&tree);
     assert(bplus_size(&tree) == 1);
     assert(!bplus_member(&tree, o1));
     assert(bplus_member(&tree, o2));

     bplus_delete(&tree, o1);
     assert(bplus_size(&tree) == 1);
     assert(!bplus_member(&tree, o1));
     assert(bplus_member(&tree, o2));

     bplus_delete(&tree, o2);
     assert(bplus_size(&tree) == 0);
     assert(!bplus_member(&tree, o1));
     assert(!bplus_member(&tree, o2));
}

int nodesize(int nodeid)
{
     struct internal *internal;
     struct leaf *leaf;
     int i;

     if (nodeid > 0) {
          internal = &tree.nodes[nodeid].internal;
          for (i = 0; i < IVALUES && internal->p[i]; i++);
          return i;
     } else {
          assert(nodeid < 0);
          leaf = &tree.nodes[-nodeid].leaf;
          for (i = 0; i < LVALUES && leaf->v[i]; i++);
          return i;
     }
}

void split_leaf(void)
{
     int i, rootid;
     struct internal *root;
     struct object *object;

     assert(bplus_size(&tree) == 0);

     rootid = tree.rootid;
     for (i = 0; i < 5; i++) {
          bplus_insert(&tree, make_object(i, i));
     }
     assert(bplus_size(&tree) == 5);
     assert(tree.rootid == rootid);
     assert(nodesize(tree.rootid) == 1);

     bplus_insert(&tree, make_object(5, 5));
     assert(bplus_size(&tree) == 6);
     assert(tree.rootid == rootid);
     assert(nodesize(tree.rootid) == 2);
     root = &tree.nodes[rootid].internal;
     assert(nodesize(root->p[0]) == 3);
     assert(nodesize(root->p[1]) == 3);

     for (i = 0; i < 6; i++) {
          assert(bplus_size(&tree) == 6 - i);
          object = (struct object *)bplus_search(&tree, i);
          assert(object);
          bplus_delete(&tree, object);
          assert(bplus_size(&tree) == 6 - i - 1);
     }

     /*
     bplus_delete(&tree, make_object(1, 1));
     assert(bplus_size(&tree) == 5);
     assert(nodesize(tree.rootid) == 1);
     */
}

void many_same_key(void)
{
     int i;
     struct object *object;
     struct object *objects[6];

     assert(bplus_size(&tree) == 0);
     for (i = 0; i < 6; i++)
          bplus_insert(&tree, make_object(i, 1));
     assert(bplus_size(&tree) == 6);
     for (i = 0, object = bplus_search(&tree, 1); 
          i < 6; 
          i++, object = bplus_search(NULL, 1)) {
          assert(object);
          objects[i] = object;
          assert(object->key == 1);
     }
     /* Now delete them */
     for (i = 0; i < 6; i++) {
          assert(bplus_size(&tree) == 6 - i);
          bplus_delete(&tree, objects[i]);
          assert(bplus_size(&tree) == 6 - i - 1);
     }
     assert(bplus_size(&tree) == 0);
}

void many_same_key_b(void)
{
     int i;
     struct object *object;
     struct object *objects[6];

     assert(bplus_size(&tree) == 0);
     for (i = 0; i < 6; i++)
          bplus_insert(&tree, make_object(i, 1));
     assert(bplus_size(&tree) == 6);
     for (i = 0, object = bplus_search(&tree, 1); 
          i < 6; 
          i++, object = bplus_search(NULL, 1)) {
          assert(object);
          objects[i] = object;
          assert(object->key == 1);
     }
     /* Now delete them */
     for (i = 0; i < 6; i++) {
          assert(bplus_size(&tree) == 6 - i);
          bplus_delete(&tree, objects[5 - i]);
          assert(bplus_size(&tree) == 6 - i - 1);
     }
     assert(bplus_size(&tree) == 0);
}

void split_root(void)
{
     int i, rootid;

     rootid = tree.rootid;
     for (i = 0; i < LVALUES * IVALUES + 1; i++)
          bplus_insert(&tree, make_object(i, i));
     assert(bplus_size(&tree) == LVALUES * IVALUES + 1);
     assert(tree.rootid != rootid);
}

#if 0
void trace(void)
{
     struct object *o;

     init_bplus(&tree, 33821, 0);
     bplus_insert(&tree, (void *)1560);
     bplus_insert(&tree, (void *)2670);
     bplus_insert(&tree, (void *)1545);
     bplus_insert(&tree, (void *)2709);
     bplus_insert(&tree, (void *)1426);
     bplus_insert(&tree, (void *)2938);
     check_leaf_linkage();

     bplus_delete(&tree, (void *)1560);
     check_leaf_linkage();
     bplus_delete(&tree, (void *)2670);
     check_leaf_linkage();

     bplus_insert(&tree, (void *)1433);
     check_leaf_linkage();
     bplus_insert(&tree, (void *)2755);                     /* BOOM! */
     check_leaf_linkage();
     bplus_insert(&tree, (void *)1291);
     check_leaf_linkage();
     bplus_insert(&tree, (void *)2862);
     check_leaf_linkage();
     bplus_insert(&tree, (void *)914);
     check_leaf_linkage();
     bplus_insert(&tree, (void *)3268);
     check_leaf_linkage();
     bplus_insert(&tree, (void *)869);
     check_leaf_linkage();
     bplus_insert(&tree, (void *)3275);
     check_leaf_linkage();
     bplus_insert(&tree, (void *)1236);
     check_leaf_linkage();
     bplus_insert(&tree, (void *)2930);
     check_leaf_linkage();
     bplus_insert(&tree, (void *)1235);
     bplus_insert(&tree, (void *)2896);
     bplus_insert(&tree, (void *)1442);
     bplus_insert(&tree, (void *)2716);
     bplus_insert(&tree, (void *)1220);
     bplus_insert(&tree, (void *)2901);
     bplus_insert(&tree, (void *)1195);
     bplus_insert(&tree, (void *)2937);
     bplus_insert(&tree, (void *)1241);
     bplus_insert(&tree, (void *)2954);
     bplus_insert(&tree, (void *)1437);
     bplus_insert(&tree, (void *)2794);
     bplus_insert(&tree, (void *)1229);
     bplus_insert(&tree, (void *)2926);
     bplus_insert(&tree, (void *)1382);
     bplus_insert(&tree, (void *)2879);
     bplus_insert(&tree, (void *)1420);
     bplus_insert(&tree, (void *)2706);
     bplus_insert(&tree, (void *)1423);
     bplus_insert(&tree, (void *)2730);
     bplus_insert(&tree, (void *)1332);
     bplus_insert(&tree, (void *)2850);
     bplus_insert(&tree, (void *)1201);
     bplus_insert(&tree, (void *)2944);
     bplus_insert(&tree, (void *)1191);
     bplus_insert(&tree, (void *)2946);
     bplus_insert(&tree, (void *)1355);
     bplus_insert(&tree, (void *)2782);
     bplus_insert(&tree, (void *)1320);
     bplus_insert(&tree, (void *)2832);
     bplus_insert(&tree, (void *)1526);
     bplus_insert(&tree, (void *)2602);
     bplus_insert(&tree, (void *)1197);
     bplus_insert(&tree, (void *)2978);
     bplus_insert(&tree, (void *)1347);
     bplus_insert(&tree, (void *)2889);
     bplus_insert(&tree, (void *)1396);
     bplus_insert(&tree, (void *)2761);
     bplus_insert(&tree, (void *)1347);
     bplus_insert(&tree, (void *)2826);
     bplus_insert(&tree, (void *)1177);
     bplus_insert(&tree, (void *)2949);
     bplus_insert(&tree, (void *)1082);
     bplus_insert(&tree, (void *)3046);
     bplus_insert(&tree, (void *)1158);
     bplus_insert(&tree, (void *)2969);
     bplus_insert(&tree, (void *)1341);
     bplus_insert(&tree, (void *)2823);
     check_leaf_linkage();
     bplus_delete(&tree, (void *)1195);
     check_leaf_linkage();
     assert(bplus_search(&tree, 1195)); /* Deleted only one, so still
                                         * one left. */
     check_leaf_linkage();
     o = bplus_search(&tree, 0);
     while (o) {
          printf("%d ", (int)o);
          o = bplus_search(NULL, 0);
     }
     printf("\n");
}

void searchproblem(void)
{
     void *o;

     init_bplus(&tree, 20, 0);
     bplus_insert(&tree, (void *)1);
     bplus_insert(&tree, (void *)1);
     bplus_insert(&tree, (void *)1);
     bplus_insert(&tree, (void *)1);
     bplus_insert(&tree, (void *)1);
     bplus_insert(&tree, (void *)1);
     bplus_insert(&tree, (void *)2);
     bplus_print_all_values(&tree);
     o = bplus_search(&tree, 1);
     while (o) {
          printf("%d ", (int)o);
          o = bplus_search(NULL, 0);
     }
     printf("\n");
     bplus_delete(&tree, (void *)2);
     bplus_print_all_values(&tree);
     bplus_delete(&tree, (void *)1);
     bplus_print_all_values(&tree);
}
#endif

int main(int argc, char *argv[])
{
     struct object o;

     init_bplus(&tree, 20, (char *)&o.key - (char *)&o);
     assert(bplus_size(&tree) == 0);

     one_node();
     split_leaf();
     many_same_key();
     many_same_key_b();
     split_root();
#if 0
     trace();
     searchproblem();
#endif
     return 0;
}

