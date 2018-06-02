 #ifndef _LINKEDLIST_H_
 #define _LINKEDLIST_H_
 
 #include <iostream>
 #include "Node.hpp"
 
 using namespace std;
 
 /*....DO NOT EDIT BELOW....*/
 template <class T> 
 class LinkedList {
     private:
 		/*First node of the linked-list*/
         Node<T>* head;
         /*Last node of the linked-list*/
 		Node<T>* tail;
 		/*size of the linked-list*/
 		size_t  size; 
     public:
 
         LinkedList();
         LinkedList(const LinkedList<T>& ll);
         LinkedList<T>& operator=(const LinkedList<T>& ll);
         ~LinkedList();
 
         /* Return head of the linked-list*/
         Node<T>* getHead() const;
         /* Set head of the linked-list*/
         void setHead(Node<T>* n);
         /* Return tail of the linked-list*/
         Node<T>* getTail() const;
         /* Set tail of the linked-list*/
         void setTail(Node<T>* n);
         /* Get the previous node of the node that contains the data item. 
          * If the head node contains the data item, this method returns NULL.*/
         Node<T>* findPrev(const T& data) const;
         /* Get the node that stores the data item. 
          * If data is not found in the list, this function returns NULL.*/
         Node<T>* findNode(const T& data) const;
         /* Insert a new node to store the data item. 
          * The new node should be placed after the “prev” node. 
          * If prev is NULL then insert new node to the head.*/
         void insertNode(Node<T>* prev, const T& data); 
         /* This method is used to delete the node that is next to “prevNode”. 
          * PS:prevNode is not the node to be deleted. 
 		 * If prev is NULL then delete head node. 
 		 */
         void deleteNode(Node<T>* prevNode);  
         /* This method is used to clear the contents of the list.*/
         void clear();
         /* This method returns true if the list empty, otherwise returns false.*/
         bool isEmpty() const;
         /* This method returns the current size of the list. */
         size_t getSize() const;
         /*Prints the list. This method was already implemented. Do not modify.*/
         void print() const;
 };
 
 template <class T>
 void LinkedList<T>::print() const{
     const Node<T>* node = head;
     while (node) {
         std::cout << node->getData();
         node = node->getNext();
     }
     cout<<std::endl;
 }
 
 /*....DO NOT EDIT ABOVE....*/
 
 /* TO-DO: method implementations below */
 
 template <class T>
 LinkedList<T>::LinkedList() : head(), tail(),size(0) {}
 
 template <class T>
 LinkedList<T>::LinkedList(const LinkedList<T>& ll) : head(ll.head), tail(ll.tail), size(ll.size) {}
 
 template <class T>
 LinkedList<T>& LinkedList<T>::operator=(const LinkedList<T>& ll)
 {
     LinkedList<T> newlist;
     newlist.head=ll.head;
     newlist.tail=ll.tail;
     newlist.size=ll.size;
     return newlist;
 }
 
 template <class T>
 LinkedList<T>::~LinkedList()
 {
     this->clear();
     delete head,tail;
 }
 
 template <class T>
 Node<T>* LinkedList<T>::getHead() const
 {
     return this->head;
 }
 
 template <class T>
 void LinkedList<T>::setHead(Node<T> *n)
 {
     this->head=n;
 }
 
 template <class T>
 Node<T>* LinkedList<T>::getTail() const
 {
     return this->tail;
 }
 
 template <class T>
 void LinkedList<T>::setTail(Node<T>* n)
 {
     this->tail=n;
 }
 
 template <class T>
 Node<T>* LinkedList<T>::findPrev(const T& data) const
 {
     if(!this->getHead() || this->getHead()->getData()==data) return NULL;
     
     Node<T> *prev;
     prev=this->getHead();
     while(prev->getNext())
     {
         if(prev->getNext()->getData()==data)
         {
             return prev;
         }
         prev=prev->getNext();
     }
     return NULL;
 }
 
 template <class T>
 Node<T>* LinkedList<T>::findNode(const T& data) const
 {
     Node<T>* index=getHead();
     while(index)
     {
         if(index->getData()==data)
         {
             return index;
         }
         index=index->getNext();
     }
     return NULL;
 }
 
 template <class T>
 void LinkedList<T>::insertNode(Node<T>* prev, const T& data)
 {
 
     Node<T> *newnode=new Node<T>(data);
     if(!prev)
     {
         if(!this->getHead())
         {
             this->setHead(newnode);
             this->setTail(newnode);
         }
         else
         {
             newnode->setNext(this->getHead());
             this->setHead(newnode);
         }
     }
     else
     {
         newnode->setNext(prev->getNext());
         prev->setNext(newnode);
         if(prev==this->getTail())
         {
             this->setTail(newnode);
         }
     }
 }
 
 template <class T>
 void LinkedList<T>::deleteNode(Node<T>* prevNode)
 {
     if(!prevNode)
     {
         if(!head) {return;} 
         Node<T> *dummy=this->getHead();
         setHead(dummy->getNext());
         //delete dummy->getData();
         delete dummy->getNext();
         delete this->getHead();
     }
     else
     {
         if(!prevNode->getNext()){ return ;}
         
         // ERROR? //
         Node<T> *dummy;
         dummy=(prevNode->getNext())->getNext();
         //delete prevNode->getNext()->getData();
         delete prevNode->getNext()->getNext();
         delete prevNode->getNext();
         prevNode->setNext(dummy);
         delete dummy;
         
         
     }
 }
 
 template <class T>
 void LinkedList<T>::clear()
 {
     if(this->getHead()) 
     {
         Node<T> *dummy,*dummy2;
         dummy2=this->getHead();
         while(dummy2)
         {
             dummy=this->getHead();
             dummy2=dummy->getNext();
             this->setHead(dummy2);
             delete dummy->getNext();
             delete dummy;
         }
         delete dummy2;
     }
     this->setHead(NULL);
     this->setTail(NULL);
 }
 
 template <class T>
 bool LinkedList<T>::isEmpty() const
 {
     if(!this->getHead()) {return 1;}
     return 0;
 }
 
 template <class T>
 size_t LinkedList<T>::getSize() const
 {
     Node<T> *dummy;
     int count=0;
     dummy=this->getHead();
     while(dummy)
     {
         dummy=dummy->getNext();
         count++;
     }
     return count;
 }
 
 
 /* end of your implementations*/
 
 #endif
 


