#include <iomanip>
#include "Miner.h"
#include "Utilizer.h"

/*
YOU MUST WRITE THE IMPLEMENTATIONS OF THE REQUESTED FUNCTIONS
IN THIS FILE. START YOUR IMPLEMENTATIONS BELOW THIS LINE 
*/





Miner::Miner(std::string name):minerName(name){};
Miner::~Miner()
{
    for(int i=0;i<this->getBlockchainCount();i++)
    {
        if(!blockchainHeads[i]->getForkType())
        {
            delete blockchainHeads[i];


            if(blockchainHeads[i]== nullptr || blockchainHeads[i]->getHead()==nullptr)
            {
                std::vector<Blockchain*>::iterator it=blockchainHeads.begin()+i-1;
                blockchainHeads.erase(it);
            }
        }
    }
}


void Miner::createNewBlockchain()
{
    //printf("A1");
    //std::vector<Blockchain*>::iterator blocksIterator=(this->blockchainHeads).end();
    if(this->blockchainHeads.size()==0)
    {
        this->blockchainHeads.push_back(new Blockchain(0));
        //printf("%d",this->blockchainHeads.back()->getHead()->getValue());
    }
    else
    {
        Blockchain *lastBlockchain=this->blockchainHeads.back();
        ////printf("A3");
        int newID=lastBlockchain->getID()+1;
        ////printf("A4");
        this->blockchainHeads.push_back(new Blockchain(newID));
        ////printf("A2");
    }

}


void Miner::mineUntil(int cycleCount)
{
    ////printf("B1");
    if(cycleCount<=0) return;
    else
    {
        for(int i=0; i<this->blockchainHeads.size(); i++)
        {
            ++(*blockchainHeads[i]);
            //printf("%ld",blockMiner->getChainLength());
            //++blockMiner;
            //printf("%d",blockMiner->getHead()->getValue());
        }


        mineUntil(cycleCount-1);
    }
    ////printf("B2");
}


void Miner::demineUntil(int cycleCount)
{
    //printf("C1");
    if(cycleCount==0) return;
    else
    {
        for(int i=0; i<this->blockchainHeads.size();i++)
        {
            --(*blockchainHeads[i]);
            //--blockMiner;
        }
        demineUntil(cycleCount-1);
    }
    //printf("C2");
}


double Miner::getTotalValue() const
{
    //printf("D1");
    double totalValue=0;
    for(int i=0; i<this->blockchainHeads.size();i++)
    {
        if(!blockchainHeads[i]->getForkType()) totalValue+=(blockchainHeads[i])->getTotalValue();
        //totalValue+=(!blockchainHeads[i]->getForkType())? 0:(*blockchainHeads[i]);
        //if(!blockMiner->getForkType()) totalValue+=blockMiner->getTotalValue();
    }
    //printf("D2");
    return totalValue;
}

long Miner::getBlockchainCount() const
{
    //printf("E1");
    return (long)this->blockchainHeads.size();
    //printf("E2");
}


Blockchain* Miner::operator[](int id) const
{
    //printf("F1");
    for(int i=0; i<this->blockchainHeads.size();i++)
    {
        Blockchain *blockMiner=this->blockchainHeads[i];
        if(blockMiner->getID()==id) return blockMiner;
    }
    return nullptr;
    //printf("F2");
}


bool Miner::softFork(int blockchainID)
{
    //printf("G1");
    Blockchain *softChain=(*this)[blockchainID];
    if(softChain== nullptr) return false;
    else
    {
        Blockchain *lastBlockchain=this->blockchainHeads.back();
        int nextAvailableID=lastBlockchain->getID()+1;
        Koin *newHead=softChain->getHead();
        if(newHead!= nullptr)
        {
            while(newHead->getNext()!= nullptr)
            {
                newHead=newHead->getNext();
            }
        }
        Blockchain *forkedChain= new Blockchain(nextAvailableID,newHead);
        forkedChain->changeForkType("soft");
        this->blockchainHeads.push_back(forkedChain);

        //printf("G2");
        return true;

    }

}

bool Miner::hardFork(int blockchainID)
{
    //printf("H1");
    //Blockchain *hardChain=(*this)[blockchainID];
    if((*this)[blockchainID]== nullptr) return false;
    else
    {
        Blockchain *lastBlockchain=(*this)[blockchainID];
        Koin *newHead=(*this)[blockchainID]->getHead();
        if((*this)[blockchainID]->getHead()!= nullptr)
        {
            while(newHead->getNext()!= nullptr)
            {
                newHead=newHead->getNext();
            }
        }
        int nextAvailableID=this->blockchainHeads.back()->getID()+1;
        Koin* forkedCoin=new Koin(newHead->getValue());

        Blockchain *forkedChain=new Blockchain(nextAvailableID,forkedCoin);
        this->blockchainHeads.push_back(forkedChain);







/*
        int nextAvailableID=this->blockchainHeads.back()->getID()+1;
        if((*this)[blockchainID]->getHead()!= nullptr)
        {
            while(newHead->getNext()!= nullptr)
            {
                newHead=newHead->getNext();
            }
        }
        //Koin *createdNewHead=;
        Blockchain *forkedChain= new Blockchain(nextAvailableID);
        forkedChain->setHead(new Koin(newHead->getValue()));
*/
        //printf("H2");
        return true;
    }

}


std::ostream& operator<<(std::ostream& os, const Miner& miner)
{
    //printf("I1");
    int precision=Utilizer::koinPrintPrecision();
    os<<std::setprecision(precision)<<std::fixed << "-- BEGIN MINER --" <<std::endl;
    if(&miner!= nullptr)
    {
        os<<"Miner name: "<<miner.minerName<<std::endl;
        os<<"Blockchain count: "<<miner.getBlockchainCount()<<std::endl;
        os<<"Total value: "<<miner.getTotalValue()<<std::endl<<std::endl;
        for(int i=0; i<miner.blockchainHeads.size();i++)
        {
            //Blockchain *blockMiner=miner.blockchainHeads[i];
            //printf("%ld", blockMiner->getChainLength());
            os<<*miner.blockchainHeads[i]<<std::endl;
        }
    }

    os<<std::endl<<"-- END MINER --"<<std::endl;
    //printf("I2");
}

