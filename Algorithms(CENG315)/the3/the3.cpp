#include <stdio.h>//debugging etc.
#include <fstream>//read file
#include <sstream>//strings-file
#include <string>
#include <vector>//corridor info etc.
#include <tuple>
#include <limits>//infinity etc.
#include <algorithm>//std::remove etc.
#include <iostream>//debugging cout etc.

//main routine that executes
int main(int argc, char const *argv[])
{
	std::vector<int> path;//path and ammo of TOTAL execution
	int totalAmmoUsed=0;//includes normalization so you should decrease by normalization*X at the end

	int temp,temp2,temp3;//used for pushing values to vectors etc.

	std::cout<<"main started"<<std::endl;
	//get file
	std::ifstream infile("the3.inp");
	// ### read file ###
	std::string line;

	//first line is ammo
	std::getline(infile,line);
	std::istringstream iss(line);
	int currAmmo;
	iss>>currAmmo;
	iss.str("");//clear iss
	iss.clear();

	//second line is total number of rooms 3<=X<=10000
	std::getline(infile,line);
	std::istringstream iss2(line);
	int roomCount;
	iss2>>roomCount;
	iss2.str("");
	iss2.clear();

	//third line is room number of bunny 
	std::getline(infile,line);
	std::istringstream iss3(line);
	int bunnyRoom;
	iss3>>bunnyRoom;
	iss3.str("");
	iss3.clear();
	
	//fourth line is the room number of key to the scientist room 
	std::getline(infile,line);
	std::istringstream iss4(line); 
	int keyToSci;
	iss4>>keyToSci;
	iss4.str("");
	iss4.clear();

	//fifth line is room number of scientist
	std::getline(infile,line);
	std::istringstream iss5(line);
	int SciRoom;
	iss5>>SciRoom;
	iss5.str("");
	iss5.clear();

	//sixth line is number of rooms locked in the odd periods
	//				followed by room numbers
	std::getline(infile,line);
	std::istringstream iss6(line);
	int lockedInOddCount;
	iss6>>lockedInOddCount;
	std::vector<int> lockedInOdd;
	for (int i = 0; i < lockedInOddCount; ++i){ iss>>temp ; lockedInOdd.push_back(temp);}
	iss6.str("");
	iss6.clear();

	//seventh line is number of rooms locked in the even periods
	//				followed by room numbers
	std::getline(infile,line);
	std::istringstream iss7(line);
	int lockedInEvenCount;
	iss7>>lockedInEvenCount;
	std::vector<int> lockedInEven;
	for (int i = 0; i < lockedInEvenCount; ++i){ iss>>temp ; lockedInEven.push_back(temp);}
	iss7.str("");
	iss7.clear();


	// eight line is number of corridors 
	std::getline(infile,line);
	std::istringstream iss8(line);
	int corridorCount;
	iss8>>corridorCount;
	
	// next number of corridor amount of lines consists of 3 integers
	//			first two are room numbers connected, last one is amount of ammo you need to use
	std::vector<std::tuple<int,int,int>> corridors;
	for (int i = 0; i < corridorCount; ++i)
	{
		iss8>>temp>>temp2>>temp3;
		corridors.push_back(std::make_tuple(temp,temp2,temp3));
	}
	iss8.str("");
	iss8.clear();


	// next line is number of rooms with ammo inside
	std::getline(infile,line);
	std::istringstream iss9(line);
	int ammoRoomCount;
	iss9>>ammoRoomCount;

	// next number of roooms with ammo lines consists two integers
	//			fist is room number, second is amount of ammo inside
	std::vector<std::pair<int,int>> ammoRooms;
	for (int i = 0; i < ammoRoomCount; ++i)
	{
		iss9>>temp>>temp2;
		ammoRooms.push_back(std::make_pair(temp,temp2));
	}
	iss9.str("");
	iss9.clear();

	std::cout<<"input read"<<std::endl;
	//close the file
	if (infile.is_open()) infile.close();
	std::cout<<"closed input file"<<std::endl;

	//first go to the bunnyRoom
	//second go to keyToSci room
	//third go to the SciRoom(dont move no more)--with minimal ammo consumption
	//assume the path is something like-->  1,...,a,...,b,...,c,...,SciRoom
	//assume ammoRooms are d,e(if they exist AND they are not already in the path)
	//after you reach SciRoom you should (if there exists ammoRooms)
	//							case:1 ammoRoom-> find the shortest path from a to (d or e) to b THEN if the path is profitable change paths with original--note that bunnyRoom and keyToSci rooms must not be between a&b since they will get erased from the path
	//							case:2 ammoRoom-> also find the shortest path from b to (d or e) to c THEN if the path is profitable change paths with original--note that bunnyRoom and keyToSci rooms must not be between b&c since they will get erased from the path
	//		new path is: dependent on ammoRoomCount&profit but the ...'s get erased between a-b and/or b-c, and nodes to and from ammoRooms get inserted there

	/* how to: find cheapest path from one node to another
	*	do dijkstra search
	*	to mitigate ammo inside ammoRooms decrease distance from any room to ammoRoom by ammo amount; 
	*		then (if there is a corridor with negative distance) normalize all corridors
	*
	*	dijkstra::
	*	::::::::::
	*	initialize dist array(distances from source node to each node in graph) by:
	*		:dist(source)=0, dist(anything else)=infinity
	*	initialize Q queue(queue of all nodes in the graph, at the end Q will be empty)
	*	initialize S set(a set to indicate visited nodes). At the end S will include all nodes
	*	initialize A array of size roomCount, to hold the pointer to shortest path vector V of a node.
	*	::::::::::
	*	While Q is not empty:
	*		:pop the node v, that is not already in S, from Q with the smallest dist(v). So in the first run source node will be chosen since only dist(s)=0<infinity
	*		:add node v to S, to indicate v has been visited
	*		:update dist calues of adjacent nodes of the current node v as follows->
	*			:for each new adjacent node u
	*				:if dist(v) + weight(u,v) < dist(u) ----> dist(u) <- dist(v)+weight(u,v)
	*
	*	do the ammo calculations:: increase the ammo by corridors*normalization :: if we visited ammo rooms, decrease the ammo by ammo in that room*(passing corridors connected to ammoRoom-1	
	*	totalNormalization is the amount we increased all corridor weights
	*
	*/
	int totalNormalization = 0;//the amount we increased all corridor weights

	//ALGORITHM:::decrease by ammoRoom ammo & normalize
	for (int i = 0; i < ammoRoomCount; i++)//for each ammoRoom
	{
		int ammoInRoom = ammoRooms[i].second;
		std::vector<int> connectedToAmmoRoomL;//room numbers of rooms that are connected to ammoRoom
		std::vector<int> connectedToAmmoRoomR;//room numbers of rooms that are connected to ammoRoom

		int minL = std::numeric_limits<int>::infinity(), minR = minL;//minimum weight of corridors
		int totalConnectedRoomsL = 0, totalConnectedRoomsR=0;//used for normalizing etc.
		for (int j = 0; j < corridorCount; j++)//for each corridor that ammoRoom is connected to
		{
			if (std::get<0>(corridors[j]) == ammoRooms[i].first)
			{
				if (minL > std::get<2>(corridors[j])) minL = std::get<2>(corridors[j]);
				connectedToAmmoRoomL.push_back(std::get<1>(corridors[j]));
				totalConnectedRoomsL++;
			}
			else if (std::get<1>(corridors[j]) == ammoRooms[i].first)
			{
				if (minR > std::get<2>(corridors[j])) minR = std::get<2>(corridors[j]);
				connectedToAmmoRoomR.push_back(std::get<0>(corridors[j]));
				totalConnectedRoomsR++;
			}
		}
		
		//find the min weight of connected corridors--save it to minL
		minL = (minL < minR ? minL : minR);
		//if there will be a negative weight corridor normalize all corridors by adding normalization to all corridors--do this before decreasing, to shorten the code
		int normalization = ammoRooms[i].second - minL;
		if (normalization > 0) 
		{
			for (int j = 0; j < corridorCount; j++) std::get<2>(corridors[i]) += normalization;
			totalNormalization += normalization;
		}

		//decrease weight of all connected corridors by ammoInRoom
		for (int j = 0; j < totalConnectedRoomsL; j++) std::get<2>(corridors[connectedToAmmoRoomL[j]]) -= minL;
		for (int j = 0; j < totalConnectedRoomsR; j++) std::get<2>(corridors[connectedToAmmoRoomR[j]]) -= minR;		
	}
	std::cout<<"calculations are done for ammoRoom @line202"<<std::endl;
	
	//ALGORITHM:::dijkstra::initialize
	int *dist = new int[roomCount+1];//distances from source room to each room in graph, takes values[1,roomCount]
	dist[1] = 0;//starting room's distance to itself is 0
	for (int i = 2; i < roomCount + 1; i++) dist[i] = std::numeric_limits<int>::infinity();//set all distances to infinity
	//i will be using vector instead of queue for Q
	std::vector<int> Q;//queue of all nodes(room numbers) in the graph, at the end Q will be empty
	Q.reserve(sizeof(int)*roomCount);
	for (int i = 1; i < roomCount + 1; i++) Q.push_back(i);
	//i will be using vector instead of "set" for S
	std::vector<int> S;//a set to indicate visited nodes
	S.reserve(sizeof(int)*roomCount);
	std::vector<int> *A = new std::vector<int>[roomCount + 1];//holds the pointer to shortest path vector V of a node, takes valeus[1,roomCount]//use emplace_back while adding
	int targetRoom = keyToSci;//first target is bunnyRoom

	
	std::cout<<"initializations are done @line219"<<std::endl;
	//ALGORITHM:::dijkstra::while loop--with target node targetRoom
	dijkstra: //GOTO label
	while (Q.size() > 0)
	{
		std::cout<<"Q.size is->"<<Q.size()<<std::endl;
		int v=0;//there is no room with number 0 so this works out
		
		//find the room v from Q, that is not already in S, with the smallest dist(v) that is not locked--and we dont have the key--
		int smallestDist = std::numeric_limits<int>::infinity();
		for (int i = 1; i < roomCount+1; i++)
		{
			if((i==SciRoom && targetRoom==keyToSci) || (i==bunnyRoom && targetRoom==SciRoom)) //cant pass/move through locked rooms
			{
				Q.erase(std::remove(Q.begin(), Q.end(), i), Q.end());//removes i from Q
				continue;
			}
			else if (dist[i] < smallestDist && (std::find(S.begin(), S.end(), i) == S.end()))//we found a new room with smallest dist that is not in S
			{
				v = i;//v=room number
				smallestDist = dist[i];//smallestDist= distance to room with number i(or v)
			}
		}
		
		std::cout<<"found next room "<<v<<" to check distances @line241"<<std::endl;
		//pop v
		//add v to S
		//update dist values of adjacent nodes of the current node v
		Q.erase(std::remove(Q.begin(), Q.end(), v), Q.end());//removes v from Q
		S.emplace_back(v);
		for (int i = 0; i < corridorCount; i++)//for each new adjacent node u THAT FITS THE LOCK MECHANISM, update distances AND path to u
		{
			bool isVLockedInEven=std::find(lockedInEven.begin(), lockedInEven.end(), v) != lockedInEven.end();//returns true if V is LockedInEven, so next node should not be locked in even
			bool isVLockedInOdd=std::find(lockedInOdd.begin(), lockedInOdd.end(), v) != lockedInOdd.end();//returns true if V is LockedInOdd, so next node should not be locked in odd
			//if dist(v) + weight(u, v) < dist(u)----> dist(u) <-dist(v) + weight(u, v) AND lock mechanism
			if (std::get<0>(corridors[i]) == v)
			{
				if((std::get<0>(corridors[i])==SciRoom && targetRoom==keyToSci) || (std::get<0>(corridors[i])==bunnyRoom && targetRoom==SciRoom)) continue; //cant pass/move through locked rooms

				if( (isVLockedInEven && std::find(lockedInEven.begin(), lockedInEven.end(), std::get<1>(corridors[i])) == lockedInEven.end())  ||  //lock mechanism
					(isVLockedInOdd && std::find(lockedInOdd.begin(), lockedInOdd.end(), std::get<1>(corridors[i])) == lockedInOdd.end() ) ) //lock mechanism
					{
						if (dist[v] + std::get<2>(corridors[i]) < dist[std::get<1>(corridors[i])])
						{
							dist[std::get<1>(corridors[i])] = dist[v] + std::get<2>(corridors[i]);//update distance to adjacent node
							//update path of adjacent node
							A[std::get<1>(corridors[i])].clear();
							A[std::get<1>(corridors[i])] = A[v];
							A[std::get<1>(corridors[i])].emplace_back(v);
						}
					}
				
			}
			else if (std::get<1>(corridors[i]) == v)
			{
				if((std::get<1>(corridors[i])==SciRoom && targetRoom==keyToSci) || (std::get<1>(corridors[i])==bunnyRoom && targetRoom==SciRoom)) continue; //cant pass/move through locked rooms
				
				if( (isVLockedInEven && std::find(lockedInEven.begin(), lockedInEven.end(), std::get<0>(corridors[i])) == lockedInEven.end())  ||  //lock mechanism
					(isVLockedInOdd && std::find(lockedInOdd.begin(), lockedInOdd.end(), std::get<0>(corridors[i])) == lockedInOdd.end() ) ) //lock mechanism
					{
						if (dist[v] + std::get<2>(corridors[i]) < dist[std::get<0>(corridors[i])])
						{
							dist[std::get<0>(corridors[i])] = dist[v] + std::get<2>(corridors[i]);//update distance to adjacent node
							//update path of adjacent node
							A[std::get<0>(corridors[i])].clear();
							A[std::get<0>(corridors[i])] = A[v];
							A[std::get<0>(corridors[i])].emplace_back(v);
						}
					}
			}
		}
		std::cout<<"calculations are done for adjacent rooms @line288"<<std::endl;
	}
	
	
	//now we have the path from source room to targetRoom saved @ A[targetRoom]--if sourceRoom==targetRoom, path is empty which is correct
	//take actions depending on which targetRoom(which step of execution we are)
	if(targetRoom==keyToSci)//first step
	{
		//add current path to targetRoom to path and increase ammoSpent
		path.insert(path.end(),A[targetRoom].begin(),A[targetRoom].end());
		totalAmmoUsed+=dist[targetRoom];

		//note that targetRoom is our startingRoom now

		//re-initialize
		for (int i = 1; i < roomCount + 1; i++) dist[i] = std::numeric_limits<int>::infinity();//set all distances to infinity
		dist[targetRoom] = 0;//starting room's distance to itself is 0

		//i will be using vector instead of queue for Q
		Q.clear();//clear Q--keeps memory but you need to re-reserve since you used remove() method
		Q.reserve(sizeof(int)*roomCount);
		for (int i = 1; i < roomCount + 1; i++) Q.push_back(i);
		//i will be using vector instead of "set" for S
		S.clear();//a set to indicate visited nodes

		for (int i = 1; i < roomCount+1; i++)//clear A's vectors
		{
			A[i].clear();//holds the pointer to shortest path vector V of a node, takes valeus[1,roomCount]//use emplace_back while adding
		}
		std::cout<<"finished first step @line317"<<std::endl;
		targetRoom=keyToSci;//second room is keyToSci
		goto dijkstra;//restart dijksta
	}
	if(targetRoom==SciRoom)//second step
	{
		//add current path to targetRoom to path and increase ammoSpent
		path.insert(path.end(),A[targetRoom].begin(),A[targetRoom].end());
		totalAmmoUsed+=dist[targetRoom];

		//note that targetRoom is our startingRoom now

		//re-initialize
		for (int i = 1; i < roomCount + 1; i++) dist[i] = std::numeric_limits<int>::infinity();//set all distances to infinity
		dist[targetRoom] = 0;//starting room's distance to itself is 0

		//i will be using vector instead of queue for Q
		Q.clear();//clear Q--keeps memory but you need to re-reserve since you used remove() method
		Q.reserve(sizeof(int)*roomCount);
		for (int i = 1; i < roomCount + 1; i++) Q.push_back(i);
		//i will be using vector instead of "set" for S
		S.clear();//a set to indicate visited nodes

		for (int i = 1; i < roomCount+1; i++)//clear A's vectors
		{
			A[i].clear();//holds the pointer to shortest path vector V of a node, takes valeus[1,roomCount]//use emplace_back while adding
		}
		std::cout<<"finished second step @line344"<<std::endl;
		targetRoom=SciRoom;//third -and final- room is sciRoom
		goto dijkstra;//restart dijksta
	}
	if(targetRoom==bunnyRoom)//third step
	{
		//add current path to targetRoom to path and increase ammoSpent
		path.insert(path.end(),A[targetRoom].begin(),A[targetRoom].end());
		totalAmmoUsed+=dist[targetRoom];

		//find total times we went to ammoRooms--we will use it for ammo calculations
		int ammoRoomVisitCount=0;
		if(ammoRoomCount==1)
		{
			totalAmmoUsed=std::count(path.begin(),path.end(),ammoRooms[0].second);
		}
		else if(ammoRoomCount==2)
		{
			totalAmmoUsed=std::count(path.begin(),path.end(),ammoRooms[0].second);
			totalAmmoUsed+=std::count(path.begin(),path.end(),ammoRooms[1].second);
		}
		//TODO error check here
		int denormalizedAmmoUsed=totalAmmoUsed-totalNormalization*path.size()+2*ammoRoomVisitCount*totalNormalization;
		currAmmo-=denormalizedAmmoUsed;
		std::cout<<"finished third step @line368"<<std::endl;
		// open a file in write mode.
   		std::ofstream outfile;
   		outfile.open("the3.out");
		outfile<<currAmmo<<std::endl;
		outfile<<path.size()<<std::endl;
		for(int i=0;i<path.size();i++) outfile<<path[i];
		if (outfile.is_open()) outfile.close();
		//TODO check if we can move through sciRoom etc. without having the key
		std::cout<<"finished outputting file @line377"<<std::endl;
		

		//free  
		free(dist);
		free(A);
	}
	
	
	
	
	//int **dist=new int*[roomCount];//2D array that stores pointers to data on distance between 2nodes, dis[i][j] gives distance between i-j
	//for (int i = 0; i < roomCount; i++) dist[i] = new int[roomCount];

	//std::tuple<int, int, int> corridorInfo;
	//for (int i = 0; i < roomCount; i++)//initialize to inf 
	//{
	//	for (int j = 0; j < roomCount; j++) dst[i][j] = std::numeric_limits<int>::infinity;
	//	dist[i][i] = 0;//set distance to itself to 0
	//} 
	//for (int i = 0; i < corridorCount; i++)//set inital distances--or ammo needed to pass that corridor
	//{
	//	corridorInfo=corridors[i];
	//	dist[std::get<0>(corridorInfo)][std::get<1>(corridorInfo)] = std::get<2>(corridorInfo);
	//}
	//for (int k = 0; k < roomCount; k++)
	//{
	//	for (int i = 0; i < roomCount; i++)
	//	{
	//		for (int j = 0; j < roomCount; j++)
	//		{
	//			if (dist[i][j] > dist[i][k] + dist[k][j]) dist[i][j] = dist[i][k] + dist[k][j];
	//		}
	//	}
	//}




	return 0;
}