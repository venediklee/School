 #include "bookstore.hpp"
 
 BookStore::BookStore( ) //implemented, do not change
 {
 }
 
 void
 BookStore::insert(const Book & book)
 {
   primaryIndex.insert(book.getISBN( ),book);
   BSTP::Iterator find_1=primaryIndex.find(book.getISBN( ));
   secondaryIndex.insert(SKey(book),&(*find_1));
   ternaryIndex.insert(SKey(book),&(*find_1));
 }
 
 void
 BookStore::remove(const std::string & isbn)
 {
     BSTP::Iterator find_1=primaryIndex.find(isbn);
     if(find_1==primaryIndex.end()){return;}
     SKey delete_this=SKey(*find_1);
     primaryIndex.remove(isbn);
     ternaryIndex.remove(delete_this);
     secondaryIndex.remove(delete_this);
   
 }
 
 void
 BookStore::remove(const std::string & title,
                   const std::string & author)
 {
     BSTP::Iterator p=primaryIndex.begin();
     while(p!=primaryIndex.end())
     {
         if((*p).getTitle() == title && (*p).getAuthor() == author) break;
         ++p;
     }
     SKey ptr((*p).getTitle(),(*p).getAuthor());
     primaryIndex.remove((*p).getISBN());
     secondaryIndex.remove(ptr);
     ternaryIndex.remove(ptr);
 }
 
 void
 BookStore::removeAllBooksWithTitle(const std::string & title)
 {
 }
 
 void
 BookStore::makeAvailable(const std::string & isbn)
 {
 }
 
 void
 BookStore::makeUnavailable(const std::string & title,
                            const std::string & author)
 {
 }
 
 void
 BookStore::updatePublisher(const std::string & author, 
                            const std::string & publisher)
 {
 }
 
 void
 BookStore::printBooksWithISBN(const std::string & isbn1,
                               const std::string & isbn2,
                               unsigned short since) const
 {
 }
 
 void
 BookStore::printBooksOfAuthor(const std::string & author,
                               const std::string & first,
                               const std::string & last) const
 {
 }
 
 void //implemented, do not change
 BookStore::printPrimarySorted( ) const
 {
   BSTP::Iterator it;
 
   for (it=primaryIndex.begin(); it != primaryIndex.end(); ++it)
   {
     std::cout << *it << std::endl;
   }
 }
 
 void //implemented, do not change
 BookStore::printSecondarySorted( ) const
 {
   BSTS::Iterator it;
 
   for (it = secondaryIndex.begin(); it != secondaryIndex.end(); ++it)
   {
     std::cout << *(*it) << std::endl;
   }
 }
 
 void //implemented, do not change
 BookStore::printTernarySorted( ) const
 {
   BSTT::Iterator it;
 
   for (it = ternaryIndex.begin(); it != ternaryIndex.end(); ++it)
   {
     std::cout << *(*it) << std::endl;
   }
 }
 
 