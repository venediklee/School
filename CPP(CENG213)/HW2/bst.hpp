 #ifndef _BIN_SEARCH_TREE_H_
 #define _BIN_SEARCH_TREE_H_
 
 #include <functional>
 #include <cstddef>
 #include <stack>
 #include <list>
 #include <ostream>
 //do not add any other library
 //modify parts as directed by assignment text & comments here
 
 template <typename Key, typename Object,
   typename Comparator = std::less<Key> >
   class BinarySearchTree
 {
 private: //do not change
   struct BinaryNode //node definition: a dependent type
   {
     Key key; //keys will be kept constant
     Object data; //objects that are referred to by keys may change
     BinaryNode * left;
     BinaryNode * right;
     size_t height; //height info should be updated per insert & delete
 
     BinaryNode(const Key &, const Object &,
       BinaryNode *, BinaryNode *, size_t = 0);
   };
 
 public: //do not change except for your own private utility functions
   class Iterator //iterator class will encapsulate the location within the BST
   {
   public:
     Iterator(); //dummy constructor for type declaration purposes
     Iterator & operator++(); //inorder increment
 
     Object & operator*();  //update data
     const Object & operator*() const; //view data
 
     bool operator==(const Iterator & rhs) const; //compare iterators
     bool operator!=(const Iterator & rhs) const; //compare iterators
 
   private:
     BinaryNode * current; //position
     const BinaryNode * root; //for error check not implemented
     std::stack<BinaryNode *> s; //will be used to conduct in order traversal if
     bool useStack; //this variable is set to true, ignored in == and !=
 
   private:
     Iterator(BinaryNode *, const BinarySearchTree &, bool = true);
     //other private utility functions can be declared by you
 
     friend class BinarySearchTree<Key, Object, Comparator>;
   };
 
 public: //do not change
   BinarySearchTree(); //empty tree
   ~BinarySearchTree(); //reclaim all dyn allocated mem
 
   void insert(const Key &, const Object &); //to insert new key,item
   void remove(const Key &); //remove the node with the key value (and also data) 
 
 public: //do not change
   Iterator find(const Key &) const; //single item
   std::list<Iterator> find(const Key &, const Key &) const;//range queries
 
   Iterator begin() const; //inorder begin
   Iterator end() const; //dummy NULL iterator
 
 public: //do not change
   int height() const; //return height of the tree
   size_t size() const; //return the number of items in the tree
   bool empty() const; //return whether the tree is empty or not
   void print(std::ostream &) const;
 
 private: //do not change
   BinaryNode * root; //designated root
   size_t nodes; //number of nodes 
   Comparator isLessThan; //function object to compare keys
 
 private:
   /* private utility functions that are implemented */
   void makeEmpty(BinaryNode * &);
 
   BinaryNode * find(const Key &, BinaryNode *) const;
   int height(BinaryNode *) const;
   void print(BinaryNode *, std::ostream &) const;
 
   template <typename T> //static utility function
   static const T & max(const T &, const T &);
 
   //balancing functions
   void rotateWithLeftChild(BinaryNode * &);
   void rotateWithRightChild(BinaryNode * &);
   void doubleWithLeftChild(BinaryNode * &);
   void doubleWithRightChild(BinaryNode * &);
 
   //you may add your own private utility functions down here
   void inserthelp(const Key &, const Object &, BinaryNode* &);
   void heightsetter(BinaryNode *&, BinaryNode *&);
 
 private: //not copiable, DO NOT IMPLEMENT or change
   BinarySearchTree(const BinarySearchTree &);
   const BinarySearchTree & operator=(const BinarySearchTree &);
 };
 
 //node constructor, implemented do not change
 template <typename K, typename O, typename C>
 BinarySearchTree<K, O, C>::BinaryNode::
 BinaryNode(const K & _k, const O & _d,
   BinaryNode * _l, BinaryNode * _r, size_t _h)
   : key(_k), data(_d), left(_l), right(_r), height(_h)
 {
 }
 
 //default constructor, implemented do not change
 template <typename K, typename O, typename C>
 BinarySearchTree<K, O, C>::BinarySearchTree()
   : root(NULL), nodes(0)
 {
 }
 
 //destructor, implemented do not change
 template <typename K, typename O, typename C>
 BinarySearchTree<K, O, C>::~BinarySearchTree()
 {
   makeEmpty(root);
 }
 
 //private utility function for destructor, do not change
 template <typename K, typename O, typename C>
 void
 BinarySearchTree<K, O, C>::makeEmpty(BinaryNode * & t)
 {
   if (t != NULL)
   {
     makeEmpty(t->left);
     makeEmpty(t->right);
     delete t;
     --nodes;
   }
 
   t = NULL;
 }
 
 
 
 ///////////////////////////////////////////////////////
 ////////////INSERTHELP/////////////////////////////////
 ///////////////////////////////////////////////////////
 template <typename K, typename O, typename C>
 void
 BinarySearchTree<K, O, C>::inserthelp(const K & k, const O & x, BinaryNode* & node)
 {
   //IF THERE IS NO ROOT
   if (!node) node = new BinaryNode(k, x, NULL, NULL, 0);
 
   else if (isLessThan(k,node->key))
   {
     inserthelp(k, x, node->left);
     if (height(node->left) - height(node->right) == 2)
     {
       if (isLessThan(k, node->left->key)) rotateWithLeftChild(node);//case 1;
       else doubleWithLeftChild(node);//case 2
     }
   }
 
   else if (isLessThan( node->key,k))
   {
     inserthelp(k, x, node->right);
     if (height(node->right) - height(node->left) == 2)
     {
       if (isLessThan(node->right->key, k)) rotateWithRightChild(node);//case4
       else doubleWithRightChild(node); //case 3
     }
   }
   else; //dont do shit??
       //std::cout<<"TEST"<<this->root<<std::endl;
   node->height = max(height(node->left), height(node->right)) + 1;
 
 }
 
 
 //public function to insert into BST, IMPLEMENT
 template <typename K, typename O, typename C>
 void
 BinarySearchTree<K, O, C>::insert(const K & k, const O & x)
 {
     if(!find(k,root)) nodes++;
   inserthelp(k, x, this->root);
     
 }
 
 //public function to remove key, IMPLEMENT
 template <typename K, typename O, typename C>
 void
 BinarySearchTree<K, O, C>::remove(const K & k)
 {
   BinaryNode* deleteme = find(k, this->root);
   //THERE IS NO DELETEME
   if (!deleteme) return;
 
     nodes--;
   //ROOT DELETION
   if (deleteme == this->root)
   {
     //IF ROOT=DELETEME HAS NO CHILD
     if (!(deleteme->right) && !(deleteme->left))
     {
       delete deleteme;
       return;
     }
 
     //IF ROOT=DELETEME HAS ONE CHILD
     else if ((!(deleteme->right) && (deleteme->left)) || ((deleteme->right) && !(deleteme->left)))
     {
       if (deleteme->right) this->root = deleteme->right;
       else this->root = deleteme->left;
 
       this->root->height = 0;
       delete deleteme;
       return;
     }
 
     //IF ROOT=DELETEME HAS TWO CHILDREN
     else
     {
       //CASE:RIGHT NODE DOESN'T HAVE LEFT NODE
       if (!(root->right->left))
       {
         root->right->left = root->left;
         root = root->right;
         root->height = max(height(root->left),height(root->right))+1;
         delete deleteme;
         heightsetter(root, root);
         return;
       }
       //CASE:RIGHT NODE HAS LEFT NODE
       else if (root->right->left)
       {
         BinaryNode* replaceme = root->right->left;
 
         //CASE:LEFT NODE HAS A LEFT NODE
         while (replaceme->left)
         {
           replaceme = replaceme->left;
         }
 
         BinaryNode* replacemefather = root->right;
         while (replacemefather->left->left)
         {
           replacemefather = replacemefather->left;
         }
 
         //CASE:REPLACEME HAS A RIGHT NODE
         if (replaceme->right)
         {
           replacemefather->left = replaceme->right;
         }
         else
         {
           replacemefather->left = NULL;
         }
         //DELETE
         root = replaceme;
         replaceme->left = deleteme->left;
         replaceme->right = deleteme->right;
         //root->height = max(height(root->left), height(root->right)) + 1;
         delete deleteme;
         heightsetter(replacemefather, root);
         return;
       }
     }
     
   }
 
   //REGULAR DELETION
   //IF THERE IS NO CHILD OF DELETEME
   if (!(deleteme->right) && !(deleteme->left))
   {
     //FIND THE FATHER OF DELETEME
     BinaryNode *deletemefather = this->root;
     while (deletemefather)
     {
       if ((deletemefather->left && (!isLessThan(k,deletemefather->left->key) && !isLessThan(deletemefather->left->key,k))) || (deletemefather->right && (!isLessThan(k,deletemefather->right->key) && !isLessThan(deletemefather->right->key,k)))) break;
       if (isLessThan(k, deletemefather->key))
       {
         deletemefather = deletemefather->left;
       }
       else if (isLessThan(deletemefather->key,k))
       {
         deletemefather = deletemefather->right;
       }
     }
     //REMOVE THE BOND WITH FATHER
     if (isLessThan(k, deletemefather->key)) deletemefather->left = NULL;
     else deletemefather->right = NULL;
 
     delete deleteme;
     
     heightsetter(deletemefather, root);
 
     return;
   }
 
   //IF DELETEME HAS ONE CHILD
   else if ((!(deleteme->right) && (deleteme->left)) || ((deleteme->right) && !(deleteme->left)))
   {
     //FIND THE CHILD OF DELETEME
     BinaryNode *deletemechild;
     if (!(deleteme->right) && (deleteme->left)) deletemechild = deleteme->left;
     else  deletemechild = deleteme->right;
 
     //FIND THE FATHER OF DELETEME
     BinaryNode *deletemefather = this->root;
     while (deletemefather)
     {
       if ((deletemefather->left && (!isLessThan(k,deletemefather->left->key) && !isLessThan(deletemefather->left->key,k))) || (deletemefather->right && (!isLessThan(k,deletemefather->right->key) && !isLessThan(deletemefather->right->key,k)))) break;
       if (isLessThan(k, deletemefather->key))
       {
         deletemefather = deletemefather->left;
       }
       else if (isLessThan(deletemefather->key, k))
       {
         deletemefather = deletemefather->right;
       }
     }
 
     //CHANGE THE POINTERS;
     if (isLessThan(k, deletemefather->key)) deletemefather->left = deletemechild;
     else deletemefather->right = deletemechild;
 
     delete deleteme;
     heightsetter(deletemefather, root);
     return;
   }
 
   //IF DELETEME HAS 2 CHILDREN
   else
   {
     //FIND THE FATHER OF DELETEME
     BinaryNode *deletemefather = this->root;
     while (deletemefather)
     {
       if ((deletemefather->left && (!isLessThan(k,deletemefather->left->key) && !isLessThan(deletemefather->left->key,k))) || (deletemefather->right && (!isLessThan(k,deletemefather->right->key) && !isLessThan(deletemefather->right->key,k)))) break;
       if (isLessThan(k, deletemefather->key))
       {
         deletemefather = deletemefather->left;
       }
       else if (isLessThan(deletemefather->key,k))
       {
         deletemefather = deletemefather->right;
       }
     }
     //REPLACE DELETEME WITH LEFTEST OF RIGHT SUBTREE & DELETE DELETEME
     if (deleteme->left && deleteme->right)
     {
       BinaryNode* replaceme = deleteme->right;
       while (replaceme->left)
       {
         replaceme = replaceme->left;
       }
       //DO THE POINTER EXCHANGES
       //CASE:DELETEME->RIGHT HAS NO LEFT CHILD
       if (!deleteme->right->left)
       {
         if (isLessThan(k, deletemefather->key)) deletemefather->left = replaceme;
         else deletemefather->right = replaceme;
         replaceme->left = deleteme->left;
 
         delete deleteme;
 
         heightsetter(replaceme, root);
 
         return;
       }
       //CASE:DELETEME->RIGHT HAS LEFT CHILD
       else
       {
         BinaryNode* replacemefather = deleteme->right;
         while (replacemefather->left->left)
         {
           replacemefather = replacemefather->left;
         }
         BinaryNode *replacemeright = replaceme->right;
         replaceme->left = deleteme->left;
         replaceme->right = deleteme->right;
         if (isLessThan( k,deletemefather->key)) deletemefather->left = replaceme;
         else deletemefather->right = replaceme;
         replacemefather->left = replacemeright;
         delete deleteme;
         heightsetter(replacemefather, root);
         return;
       }
       
     }
   }
 }
 template <typename K, typename O, typename C>
 void
 BinarySearchTree<K, O, C>::heightsetter(BinaryNode *& deletedfather, BinaryNode *& root)
 {
   
 
   //IF DELETEDFATHER==ROOT
   if (deletedfather == root)
   {
     deletedfather->height = max(height(deletedfather->left), height(deletedfather->right)) + 1;
     //BALANCE
     int balance = height(root->right) - height(root->left);
     int rightbalance = 0;
     if (root->right) rightbalance = height(root->right->right) - height(root->right->left);
     int leftbalance = 0;
     if (root->left) leftbalance = height(root->left->right) - height(root->left->left);
     if (balance > 1)
     {
       if (rightbalance <= -1)
       {
         doubleWithRightChild(root);
 
       }
       else rotateWithRightChild(root);
     }
     else if (balance < -1)
     {
       if (leftbalance >= 1)
       {
         doubleWithLeftChild(root);
       }
       else rotateWithLeftChild(root);
     }
     return;
   }
 
   const K & k = deletedfather->key;
   BinaryNode* traverser = root,*traverserfather;
 
   while (!(!isLessThan(k,traverser->key) && !isLessThan(traverser->key,k)))
   {
     traverserfather = traverser;
     if (isLessThan(k, traverser->key)) traverser = traverser->left;
     else traverser = traverser->right;
   }
 
   traverser->height = max(height(traverser->left), height(traverser->right)) + 1;
 
   //BALANCE STUFF
   int balance = height(traverser->right) - height(traverser->left);
   int rightbalance = 0;
   if (traverser->right) rightbalance = height(traverser->right->right) - height(traverser->right->left);
   int leftbalance = 0;
   if (traverser->left) leftbalance = height(traverser->left->right) - height(traverser->left->left);
   if (balance > 1)
   {
     if (rightbalance <= -1)
     {
       BinaryNode *traverserright = traverser->right, *traverserrightleft = traverser->right->left;
       rotateWithLeftChild(traverser->right);
       traverser->right = traverserrightleft;
       traverserfather->right = traverser->right;
       rotateWithRightChild(traverser);  
     }
     else
     {
       traverserfather->right = traverser->right;
       rotateWithRightChild(traverser);
     }
   }
   else if (balance < -1)
   {
     if (leftbalance >= 1)
     {
       BinaryNode *traverserleft = traverser->left, *traverserleftright = traverser->left->right;
       rotateWithRightChild(traverser->left);
       traverser->left = traverserleftright;
       traverserfather->left = traverser->left;
       rotateWithLeftChild(traverser);
     }
     else
     {
       traverserfather->left = traverser->left;
       rotateWithLeftChild(traverser);
     }
   }
   heightsetter(traverserfather, root);
 }
 
 
 //public function to search elements, do not change
 template <typename K, typename O, typename C>
 typename BinarySearchTree<K, O, C>::Iterator
 BinarySearchTree<K, O, C>::find(const K & key) const
 {
   BinaryNode * node = find(key, root);
 
   if (node == NULL)
   {
     return end();
   }
   else
   {               //not inorder iterator
     return Iterator(node, *this, false);
   }
 }
 
 /*
 * private utility function to search elements
 * do not change
 */
 template <typename K, typename O, typename C>
 typename BinarySearchTree<K, O, C>::BinaryNode *
 BinarySearchTree<K, O, C>::find(const K & key, BinaryNode * t) const
 {
   if (t == NULL)
   {
     return NULL;
   }
   else if (isLessThan(key, t->key))
   {
     return find(key, t->left);
   }
   else if (isLessThan(t->key, key))
   {
     return find(key, t->right);
   }
   else //found
   {
     return t;
   }
 }
 
 //range queries those within range are inserted to the list
 //IMPLEMENT
 template <typename K, typename O, typename C>
 std::list<typename BinarySearchTree<K, O, C>::Iterator>
 BinarySearchTree<K, O, C>::find(const K & lower, const K & upper) const
 {
     Iterator forstack = Iterator(root,*this,true);
     std::list<Iterator> betweens;
 
     while(this->root && forstack.current)
     {
         if(!isLessThan(forstack.current->key,lower) && !isLessThan(upper,forstack.current->key))
         {
             betweens.push_back(find(forstack.current->key));
         }
         ++forstack;
     }
     return betweens;
 }
 
 //INORDER iterator begins at ++root, do not change
 template <typename K, typename O, typename C>
 typename BinarySearchTree<K, O, C>::Iterator
 BinarySearchTree<K, O, C>::begin() const
 {
   return Iterator(root, *this);
 }
 
 //no more increment after end() also
 //returned in case of unsuccessful search
 //or when no more applicance of ++ is possible
 //do not change
 template <typename K, typename O, typename C>
 typename BinarySearchTree<K, O, C>::Iterator
 BinarySearchTree<K, O, C>::end() const
 {
   return Iterator(NULL, *this);
 }
 
 //public function to return height, do not change
 template <typename K, typename O, typename C>
 int
 BinarySearchTree<K, O, C>::height() const
 {
   return height(root);
 }
 
 /* private utility function for computing height */
 //do not change
 template <typename K, typename O, typename C>
 int
 BinarySearchTree<K, O, C>::height(BinaryNode * t) const
 {
   return (t == NULL) ? -1 : t->height;
 }
 
 //public function to return number of nodes in the tree
 //do not change
 template <typename K, typename O, typename C>
 size_t
 BinarySearchTree<K, O, C>::size() const
 {
   return nodes;
 }
 
 //public true if empty false o.w.
 //do not change
 template <typename K, typename O, typename C>
 bool
 BinarySearchTree<K, O, C>::empty() const
 {
   return nodes == 0;
 }
 
 //public function to print keys inorder to some ostream
 //do not change
 template <typename K, typename O, typename C>
 void
 BinarySearchTree<K, O, C>::print(std::ostream & out) const
 {
   print(root, out);
   out << '\n';
 }
 
 /* private utility function to print, do not change */
 template <typename K, typename O, typename C>
 void
 BinarySearchTree<K, O, C>::print(BinaryNode * t, std::ostream & out) const
 {
   if (t != NULL && t->left != NULL)
   {
     out << '[';
     print(t->left, out);
   }
   else if (t != NULL && t->left == NULL && t->right != NULL)
   {
     out << "[";
   }
 
   if (t != NULL)
   {
     if (t->left == NULL && t->right == NULL)
     {
       out << '(' << (t->key) << ')';
     }
     else if (t->left != NULL || t->right != NULL)
     {
       out << '{' << (t->key) << ",H" << t->height << '}';
     }
   }
 
   if (t != NULL && t->right != NULL)
   {
     print(t->right, out);
     out << ']';
   }
   else if (t != NULL && t->left != NULL && t->right == NULL)
   {
     out << "]";
   }
 }
 
 /* static function to compute maximum of two elements */
 //do not change
 template <typename K, typename O, typename C>
 template <typename T>
 const T &
 BinarySearchTree<K, O, C>::max(const T & el1, const T & el2)
 {
   return el1 > el2 ? el1 : el2;
 }
 
 /* ROTATIONS, do not change */
 template <typename K, typename O, typename C>
 void
 BinarySearchTree<K, O, C>::rotateWithLeftChild(BinaryNode * & k2)
 {
   BinaryNode *k1 = k2->left;
   k2->left = k1->right;
   k1->right = k2;
   k2->height = BinarySearchTree<K, O, C>::max(height(k2->left),
     height(k2->right)) + 1;
   k1->height = BinarySearchTree<K, O, C>::max(height(k1->left),
     height(k2)) + 1;
   k2 = k1;
 }
 
 template <typename K, typename O, typename C>
 void
 BinarySearchTree<K, O, C>::rotateWithRightChild(BinaryNode * & k1)
 {
   BinaryNode *k2 = k1->right;
   k1->right = k2->left;
   k2->left = k1;
   k1->height = BinarySearchTree<K, O, C>::max(height(k1->left),
     height(k1->right)) + 1;
   k2->height = BinarySearchTree<K, O, C>::max(height(k2->right),
     height(k1)) + 1;
   k1 = k2;
 }
 
 template <typename K, typename O, typename C>
 void
 BinarySearchTree<K, O, C>::doubleWithLeftChild(BinaryNode * & k3)
 {
   rotateWithRightChild(k3->left);
   rotateWithLeftChild(k3);
 }
 
 template <typename K, typename O, typename C>
 void
 BinarySearchTree<K, O, C>::doubleWithRightChild(BinaryNode * & k1)
 {
   rotateWithLeftChild(k1->right);
   rotateWithRightChild(k1);
 }
 
 /* ITERATOR functions */
 
 // dummy constructor, do not change
 template <typename K, typename O, typename C>
 BinarySearchTree<K, O, C>::Iterator::Iterator()
   : current(NULL), root(NULL), useStack(false)
 {
 }
 
 // dereferencing operator non-const version, IMPLEMENT
 template <typename K, typename O, typename C>
 O &
 BinarySearchTree<K, O, C>::Iterator::operator*()
 {
   return current->data;
 }
 
 // dereferencing operator const version, IMPLEMENT
 template <typename K, typename O, typename C>
 const O &
 BinarySearchTree<K, O, C>::Iterator::operator*() const
 {
   return current->data;
 }
 
 // compare Iterators ignoring useStack var, do not change
 template <typename K, typename O, typename C>
 bool
 BinarySearchTree<K, O, C>::Iterator::
 operator==(const Iterator & rhs) const
 {
   return current == rhs.current &&
     root == rhs.root;
 }
 
 // compare Iterators ignoring useStack var, do not change
 template <typename K, typename O, typename C>
 bool
 BinarySearchTree<K, O, C>::Iterator::
 operator!=(const Iterator & rhs) const
 {
   return !(*this == rhs);
 }
 
 // increment Iterator to point to the inorder next
 // node of then-current node, in case that no further
 // advances are possible return an Iterator that is
 // equal to end( ) , IMPLEMENT
 template <typename K, typename O, typename C>
 typename BinarySearchTree<K, O, C>::Iterator &
 BinarySearchTree<K, O, C>::Iterator::operator++()
 {
   if (!useStack) current = NULL;
   else
   {
     //FIND THE INORDER SUCCESSOR OF CURRENT AND UPDATE CURRENT
     if (!current) return *this;
     if (current->right)
     {
       current = current->right;
       s.pop();
       while (current)
       {
         s.push(current);
         if(current->left) current = current->left;
         else break;
       }
     }
     else
     {
       //IF THERE IS NO SUCCESSOR OF CURRENT
       if (s.empty()) current = NULL;
       //IF THERE IS A SUCCESSOR OF CURRENT UPDATE CURRENT AND STACK
       else
       {
         if (current != s.top())
         {
           current = s.top();
           s.pop();
         }
         else 
         {
           s.pop();
           if(!s.empty()) current = s.top();
           else current = NULL;
         }
       }
     }
   }
   return *this;
 }
 
 /* real Iterator constructor will be invoked by
 * BST member function only. if no inorder iterator
 * is required by the computation designer should
 * explicitly set useStack variable to false, o.w.
 * it will be assumed to be true. IMPLEMENT
 */
 template <typename K, typename O, typename C>
 BinarySearchTree<K, O, C>::Iterator::
 Iterator(BinaryNode * p, const BinarySearchTree & rhs, bool stk)
 {
   current = p;
   this->useStack = stk;
   this->root = rhs.root;
 
   if (stk)
   {
     if (current)
     {
       s.push(current);
       while (current->left)
       {
         current = current->left;
         s.push(current);
       }
     }
   }
 }
 
 
 #endif
 