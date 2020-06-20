//#include <pthread.h>
#include "writeOutput.h"

MinerInfo minerInfo;

int main()
{
	
}


void Miner(unsigned int ID, OreType OreType,unsigned int capacity, unsigned int interval, unsigned int totalOre)
{
	int currentOreCount = 0;
	//unsigned int ID, OreType oreType, unsigned int capacity, unsigned int current_count) {
	
	
	FillMinerInfo(&minerInfo, ID, OreType, capacity, currentOreCount);
	WriteOutput(&minerInfo, NULL, NULL, NULL, MINER_CREATED);
	
	while(capa)


}
