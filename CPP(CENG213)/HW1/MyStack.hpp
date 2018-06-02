 #ifndef MYSTACK_HPP
 #define MYSTACK_HPP
 #include "Node.hpp"
 
 /*You are free to add private/public members inside this template..*/
 template <class T>
 class MyStack{
     private:
     Node<T> *top;                
     public:
     /*Default constructor*/
     MyStack();                     					
     /*copy constructor*/
     MyStack(const MyStack<T>& rhs);  				
     /*destructor*/
     ~MyStack(); 
     /*overloaded = operator*/
     MyStack<T>& operator=(const MyStack<T>& rhs);  	
     /*returns true if stack is empty*/
     bool isEmpty() const;
     /*push newItem to stack*/
     void push(const T& newItem);
     /*pop item from stack*/
     void pop();
     /*return top item of the stack*/
     Node<T>* Top() const;
 	/*Prints the stack entries. This method was already implemented. Do not modify.*/
     void print() const;
 
 
 };
 
 template <class T>
 void MyStack<T>::print() const{
     const Node<T>* node = top;
     while (node) {
         std::cout << node->getData();
         node = node->getNext();
     }
     cout<<std::endl;
 }
 
 /* TO-DO: method implementations below */
 
 template <class T>
 MyStack<T>::MyStack() : top() {}
 
 template <class T>
 MyStack<T>::MyStack(const MyStack<T>& rhs)
 {
     this->top=NULL;
     *this=rhs;
 }
 
 template <class T>
 MyStack<T>::~MyStack()
 {
     while(!this->isEmpty()) this->pop();
     delete top; 
 }
 
 template <class T>
 MyStack<T>& MyStack<T>::operator=(const MyStack<T>& rhs)
 {
     if(!(this==&rhs))
     {
         while(!this->isEmpty()) this->pop();
         if(rhs.isEmpty())
         {
             this->top=NULL; return *this;
         }
         else
         {
             Node<T> *newnode= new Node<T>(rhs.Top()->getData());
             Node<T> *rightindex=rhs.Top()->getNext();
             Node<T> *leftindex=newnode;
             while(rightindex)
             {
                 leftindex->setNext(new Node<T>(rightindex->getData() ) );
                 leftindex=leftindex->getNext();
                 rightindex=rightindex->getNext();
             }
             leftindex->setNext(NULL);
             this->top=newnode;
             delete rightindex,leftindex,newnode;
             return *this;
         }
     }
     
     return *this;
 }
 
 
 template <class T>
 bool MyStack<T>::isEmpty() const
 {
     return !this->Top();
 }
 
 template <class T>
 void MyStack<T>::push(const T& newItem)
 {
     Node<T> *newnode=new Node<T>(newItem);
     newnode->setNext(this->Top());
     this->top=newnode;
 }
 
 template <class T>
 void MyStack<T>::pop()
 {
     Node<T> *newnode=this->Top();
     if(newnode)
     {
         this->top=this->Top()->getNext();
         delete newnode;
     }
 }
 
 template <class T>
 Node<T>* MyStack<T>::Top() const
 {
     return this->top;
 }
 
 
 #endif /* MYSTACK_HPP */
 
 
