    1 #ifndef _ARTICLE_H
    2 #define _ARTICLE_H
    3 #define EMPTY_INDEX -1
    4 #define MARKED_INDEX -99
    5 #define EMPTY_KEY ""
    6 #define MAX_LOAD_FACTOR 0.65
    7 #include <iostream>
    8 #include <string>
    9 #include <vector>
   10 #include <utility> // for pair
   11 #include <fstream> // for reading words from file
   12 
   13 class Article
   14 {
   15 public: 
   16 	// DONT CHANGE PUBLIC PART
   17 	Article( int table_size,int h1_param, int h2_param );
   18 	~Article();
   19 
   20 	int get( std::string key, int nth, std::vector<int> &path ) const;
   21 	int insert( std::string key, int original_index );
   22 	int remove( std::string key, int nth );
   23 
   24 	double getLoadFactor() const;
   25 	void getAllWordsFromFile( std::string filepath );
   26 
   27 	void printTable() const;
   28 	// DONT CHANGE PUBLIC PART
   29 private:
   30 	// YOU CAN ADD PRIVATE MEMBERS AND VARIABLES TO THE PRIVATE PART
   31 	int Popcount(unsigned int value) const;
   32 
   33 
   34 
   35 
   36 	std::pair<std::string, int>* table;
   37 
   38 	int n; // Current number of the existing entries in hash table
   39 	int table_size;
   40 	int h1_param; 
   41 	int h2_param;
   42 
   43 	void expand_table();
   44 	int hash_function( std::string& key, int i) const;
   45 	int h1( int key ) const;
   46 	int h2( int key ) const;
   47 	
   48 	int convertStrToInt( const std::string &key ) const;
   49 
   50 	bool isPrime(int n) const;
   51 	int nextPrimeAfter(int n) const;
   52 	int firstPrimeBefore(int n) const;
   53 	// YOU CAN ADD PRIVATE MEMBERS AND VARIABLES TO THE PRIVATE PART
   54 };
   55 #endif
   56 
