
 #include <iostream>
 #include <fstream>
 #include <sstream>
 #include <string>
 #include "Playlist.hpp"
 #include "Entry.hpp"
 
 using namespace std;
 
 
 Playlist::Playlist()
 {
     srand(15);
 }
 
 int Playlist::getRandomNumber(int i, int n) const
 {
     int range=n-i;
     int random = rand() % range + i;
     return random;
 }
 
 void Playlist::print()
 {
     cout << "[PLAYLIST SIZE=" << entries.getSize() <<"]";
     entries.print();
 }
 void Playlist::printHistory()
 {
     cout<<"[HISTORY]";
     history.print();
 }
 
 /* TO-DO: method implementations below */
 
 void Playlist::load(std::string fileName)
 {
     ifstream file;
     string ftitle,fgenre,fyear,newline;
     file.open(fileName.c_str());
     if(file.good())
     {
         while(1)
         {
             if(!file.good()) break;
             getline(file,newline);
             ftitle=newline.substr(0,newline.find(";") );
             fyear=newline.substr(newline.find_last_of(";")+1);
             fgenre=newline.substr(newline.find(";")+1 ,newline.length()-fyear.length()-ftitle.length()-2);
             //getline.(file,ftitle,';');
             //getline.(file,fgenre,';');
             //getline.(file,fyear,';');
             
             Entry newentry(ftitle,fgenre,fyear);
             this->insertEntry(newentry);
         }
     }
     file.close();
 }
 
 
 
 void Playlist::insertEntry(const Entry &e)
 {
     this->entries.insertNode(entries.getTail(),e);
     this->entries.setTail(this->entries.findNode(e));
     //cout<<(this->entries.getTail()==this->entries.findNode(e))<<endl;;
     HistoryRecord newentry(INSERT,e);
     this->history.push(newentry);
 }
 
