 #include "Article.h"
 
 /*#############################
 #               NOTE!         #
 #    Please don't forget to   #
 #     check PDF. Fuctions     #
 #    may have more detailed   #
 #     tasks which are not     #
 #       mentioned here.       #
 #############################*/
 
 Article::Article( int table_size,int h1_param, int h2_param )
 {    
     /*#############################
     #            TO-DO            #
     # Write a default constructor #
     #   for the Article Class     #
     #############################*/
 	this->n = 0;
 	this->table_size = table_size;
 	this->h1_param = h1_param;
 	this->h2_param = h2_param;
 
 	table=new std::pair<std::string, int> [table_size];
 	for (int i = 0; i < table_size; i++)
 	{
 		table[i].first = EMPTY_KEY;
 		table[i].second = EMPTY_INDEX;
 	}
 }
 
 Article::~Article()
 {
     /*#############################
     #             TO-DO           #
     #  Write a deconstructor for  #
     #     the Article Class       #
     #############################*/
 	
 	
 	delete [] table;
 	table_size=0;
 	n = 0;
 	h1_param = 0;
 	h2_param = 0;
 
 }
 
 int Article::get( std::string key, int nth, std::vector<int> &path ) const
 {
     /*#############################################
     #                    TO-DO                    #
     #      Your get function should return        #
     # the original index (index in the text file) #
     #    of the nth 'key' in the hash table.      #
     #    If there is no such a key, return -1     #
     #    If there is, return the original index   #
     #     In both cases, you need to push each    #
     #          visited index while probing        #
     #           that you calculated until         #
     #      finding the key, to the path vector.   #
     #############################################*/
 	
 	int i = 0, index = hash_function(key, 0), count = 0;
 
 	while (table[index].second!=EMPTY_INDEX)
 	{
 		
 		if (table[index].first == key)
 		{
 			count++;
 			if (count == nth)
 			{
 				//path.pop_back();
 				return table[index].second;
 			}
 			if (count > nth) break;
 		}
 		i++;
 		index = hash_function(key, i);
 		if (index >= table_size || i>table_size) break;
 		if(hash_function(key,0)!=index) path.push_back(index);
 	}
     return -1;
 }
 
 int Article::insert( std::string key, int original_index )
 {
     /*#########################################
     #                 TO-DO                   #
     #      Insert the given key, with the     #
     # original index value (at the text file) #
     #           to the hash table.            #
     #  Return the total number of probes you  #
     #      encountered while inserting.       #
     #########################################*/
 
 	if (getLoadFactor() > MAX_LOAD_FACTOR)
 	{
 		expand_table();
 	}
 
 	int index=hash_function(key,0),i=0;
 
 	while (table[index].first != EMPTY_KEY)
 	{
 		if (table[index].first == key && table[index].second > original_index)
 		{
 			int upper = table[index].second;
 			table[index].second = original_index;
 			original_index = upper;
 		}
 		i++;
 		index = hash_function(key, i);
 	}
 	n++;
 	table[index].first = key;
 	table[index].second = original_index;
 	return i;
 
     //return 0;
 }
 
 
 int Article::remove( std::string key, int nth )
 {
     /*#########################################
     #                  TO-DO                  #
     #      Remove the nth key at the hash     #
     #                  table.                 #
     #  Return the total number of probes you  #
     #      encountered while inserting.       #
     #   If there is no such a key, return -1  #
     #     If there, put a mark to the table   #
     #########################################*/
 
 	std::vector<int> path2;
 	if (get(key, nth,path2) == -1) return -1;
 
 	int index = hash_function(key, 0), probes = 0,i=0;
 	while (table[index].first != EMPTY_KEY || table[index].second == MARKED_INDEX)
 	{
 		if (table[index].first == key)
 		{
 			probes++;
 			if (probes == nth) break;
 		}
 		i++;
 		index = hash_function(key, i);
 		if (index > table_size) return -1;
 	}
 
 	table[index].first = EMPTY_KEY;
 	table[index].second = MARKED_INDEX;
 	n--;
 	return i ;
 
     //return i;
 }
 
 double Article::getLoadFactor() const
 {
     /*#########################################
     #                TO-DO                    #
     #      Return the load factor of the      #
     #                table                    #
     #########################################*/
 
 
 	if (this->table) return double(n) / double(table_size);
 
     return 0;
 }
 
 void Article::getAllWordsFromFile(std::string filepath)
 {
 	/*#########################################
 	#                  TO-DO                  #
 	#       Parse the words from the file     #
 	#      word by word and insert them to    #
 	#                hash table.              #
 	#   For your own inputs, you can use the  #
 	#  'inputify.py' script to fix them to    #
 	#            the correct format.          #
 	#########################################*/
 
 
 
 	int indexcount = 0;
 	std::ifstream file;
 	file.open(filepath.c_str());
 	std::string word;
 
 	while (file >> word) {
 		insert(word, ++indexcount);
 	}
 
 	file.close();
 
 }
 
 
 void Article::expand_table()
 {
     /*#########################################
     #                  TO-DO                  #
     #   Implement the expand table function   #
     #   in order to increase the table size   #
     #   to the very first prime number after  #
     #      the value of (2*table size).       #
     #         Re-hash all the data.           #
     #       Update h2_param accordingly.      #
     #########################################*/
 	
 	int oldtablesize = table_size;
 	this->table_size = nextPrimeAfter(2 * (this->table_size));
 	h2_param = firstPrimeBefore(table_size);
 
 
 	std::pair<std::string, int> *oldtable;
 	oldtable = table;
 	table= new std::pair<std::string, int>[table_size];
 	n = 0;
 	for (int i = 0; i < table_size; i++)
 	{
 		table[i].first = EMPTY_KEY;
 		table[i].second = EMPTY_INDEX;
 	}
 	for (int i = 0; i < oldtablesize; i++)
 	{
 		if(oldtable[i].first!=EMPTY_KEY) this->insert(oldtable[i].first, oldtable[i].second);
 	}
 	delete[] oldtable;
 	
 }
 
 
 int Article::hash_function( std::string& key, int i ) const
 {
     /*#########################################
     #                TO-DO                    #
     #       Implement the main hashing        #
     #    function. Firstly, convert key to    #
     #    integer as stated in the homework    #
     #      text. Then implement the double    #
     #            hashing function.            #
     #      use convertStrToInt function to    #
     #      convert key to a integer for       #
     #         using it on h1 and h2           #
     #               reminder:                 #
     #            you should return            #
     #    ( h1(keyToInt) + i*h2(keyToınt) )    #
     #            modulo table_size            #
     #########################################*/
 
 	int stringkey = convertStrToInt(key);
 	return (h1(stringkey) + i * h2(stringkey)) % table_size;
 
 
 
 
 
 	//i = 0;
 	//int totalvalue = (h1(stringkey) + i * h2(stringkey))%table_size,repeatingvalue=(h1(stringkey)+h2(stringkey)) % table_size,firsttime=0;
 	//while (table[totalvalue].first!=EMPTY_KEY)
 	//{
 	//	i++;
 	//	totalvalue = h1(stringkey) + i * h2(stringkey);
 	//	if (totalvalue == repeatingvalue)
 	//	{
 	//		if (firsttime == 0) firsttime++;
 	//		else
 	//		{
 	//			///////////////////////////////////////////////////////////
 	//			//EXPAND THE TABLE HERE
 	//			//this->expand_table();
 	//			firsttime = 0;
 	//			i--;
 	//		}
 	//	}
 	//}
 	//table[totalvalue].first = key;
 	//table[totalvalue].second = i+1;
 
 
 
 
     //return 0;
 }
 
 int Article::h1( int key ) const
 {
     /*###############################
     #              TO-DO            #
     #      First Hash function      #
     # Don't forget to use h1_param. #
     ###############################*/
 	int notkey = int(key);
 	int popcofkey = Popcount(notkey);
 	
 	return popcofkey * h1_param;
 
     //return 0;
 }
 
 int Article::h2( int key ) const
 {
     /*###############################
     #              TO-DO            #
     #     Second Hash function      #
     # Don't forget to use h2_param. #
     ###############################*/
 
 	return h2_param - (key % h2_param);
 
     //return 0;
 }
 
 
 
 
 
 
 
 int Article::Popcount(unsigned int value) const
 {
 	int totalone;
 	for (totalone = 0; value != 0; totalone++, value &= value - 1);
 	return totalone;
 }
