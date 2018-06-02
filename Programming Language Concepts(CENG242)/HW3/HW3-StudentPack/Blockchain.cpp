#include "Blockchain.h"

/*
YOU MUST WRITE THE IMPLEMENTATIONS OF THE REQUESTED FUNCTIONS
IN THIS FILE. START YOUR IMPLEMENTATIONS BELOW THIS LINE
*/

#include "Utilizer.h"
#include "iomanip"

Blockchain::Blockchain(int id):id(id),head(nullptr),softFork(false){};

Blockchain::Blockchain(int id, Koin *head): id(id), head(head),softFork(false){};

Blockchain::Blockchain(const Blockchain& rhs):id(rhs.getID()),softFork(false)
{
    //this->id=rhs.getID();
    if(rhs.head!= nullptr)
    {
        this->head=new Koin(rhs.head->getValue());
        //copy the rest?
        Koin *copyCoin=rhs.getHead()->getNext();
        Koin *newCopyCoin=this->head;
        while(copyCoin!= nullptr)
        {
            //printf("!A");
            newCopyCoin->setNext(copyCoin);
            copyCoin=copyCoin->getNext();
            newCopyCoin=newCopyCoin->getNext();
        }
    }
    else
    {
        ////////////////////////////////////////////////FIX THIS MEMORY LEAK
        //(this->head);
        this->head=nullptr;
    }
}

Blockchain& Blockchain::operator=(Blockchain&& rhs) noexcept
{
    //free lhs
    ////////////////////////////////////////error?
    Koin *deleteCoin=this->getHead(),*deleteCoinNext;
    while(deleteCoin!= nullptr)
    {
        //printf("!B");
        deleteCoinNext=deleteCoin->getNext();
        delete deleteCoin;
        deleteCoin=deleteCoinNext;
    }


    //copy rhs to lhs
    Koin *moveCoin=rhs.getHead(),*moveCoinLHS;
    //this->head=moveCoin;
    
    if(moveCoin!=nullptr) this->head= new Koin(moveCoin->getValue());
    moveCoinLHS=this->head;
    while(moveCoin->getNext()!= nullptr)
    {

        //printf("!C");
        moveCoinLHS->setNext(new Koin(moveCoin->getValue()));
        moveCoin=moveCoin->getNext();
        moveCoinLHS=moveCoinLHS->getNext();
    }

    //free rhs
    deleteCoin=rhs.getHead();
    while(deleteCoin!= nullptr)
    {
        //printf("!D");
        deleteCoinNext=deleteCoin->getNext();
        delete deleteCoin;
        deleteCoin=deleteCoinNext;
    }

    return *this;
}

//Blockchain& Blockchain::operator=(const Blockchain& rhs) = delete;
Blockchain::~Blockchain()
{
    //free lhs
    ////////////////////////////////////////error?
    Koin *deleteCoin=this->getHead(),*deleteCoinNext;
    while(deleteCoin!= nullptr)
    {
        //printf("!E");
        deleteCoinNext=deleteCoin->getNext();
        delete deleteCoin;
        deleteCoin=deleteCoinNext;
    }
}

int Blockchain::getID() const
{
    if(this== nullptr) return -1;
    return this->id;
}
Koin* Blockchain::getHead() const
{
    if(this== nullptr) return nullptr;
    return this->head;
}

bool Blockchain::getForkType() const
{
    if(this== nullptr) return false;
    return this->softFork;
}

double Blockchain::getTotalValue() const
{
    Koin *valueCoin=this->getHead();
    double totalValue=0;
    while(valueCoin!= nullptr)
    {
        //printf("!F");
        totalValue+=valueCoin->getValue();
        valueCoin=valueCoin->getNext();
    }
    return totalValue;
}


long Blockchain::getChainLength() const
{
    Koin *countCoin=this->getHead();
    int chainLength=0;
    while(countCoin!= nullptr)
    {
        //printf("!G");
        chainLength++;
        countCoin=countCoin->getNext();
    }
    return chainLength;
}


void Blockchain::operator++()
{
    if(this->getChainLength()==0) this->head=new Koin(Utilizer::fetchRandomValue());
    else
    {
        Koin *endOfChain=this->getHead();
        while(endOfChain->getNext()!= nullptr)
        {
            //printf("!H");
            endOfChain=endOfChain->getNext();
        }
        endOfChain->setNext(new Koin(Utilizer::fetchRandomValue()));
    }
    //Koin addCoin = new Koin(Utilizer::fetchRandomValue());
}

///////////////////////////IMPLEMENT
void Blockchain::operator--()
{

    Koin *endOfChain,*oneBeforeEndOfChain;
    if(this->getHead()==nullptr) return;
    else if(this->getChainLength()==1 && this->getForkType() ) return;
    else if(this->getChainLength()==1 && !(this->getForkType()))
    {
       Koin *endOfChain=this->head;
       this->head=nullptr; 
       delete endOfChain;
    } 
    else
    {
        endOfChain=this->getHead()->getNext();
        oneBeforeEndOfChain=this->getHead();
        while(endOfChain->getNext()!=nullptr)
        {
            oneBeforeEndOfChain=oneBeforeEndOfChain->getNext();
            endOfChain=endOfChain->getNext();

        }
        oneBeforeEndOfChain->setNext(nullptr);
        delete endOfChain;
    }
}



Blockchain& Blockchain::operator*=(int multiplier)
{
    Koin *multiplyCoin=this->getHead();
    while(multiplyCoin!= nullptr)
    {
        multiplyCoin->KoinMultiplier(multiplier);
        multiplyCoin=multiplyCoin->getNext();
    }
    /*
    Koin *multiplyCoin=this->getHead();
    Koin *multipliedCoin;
    bool isItFirst=true;
    while(multiplyCoin!= nullptr)
    {
        //printf("!J");
        multipliedCoin= new Koin((multiplyCoin->getValue())*multiplier);
        multipliedCoin->setNext(multiplyCoin->getNext());
        multiplyCoin=multiplyCoin->getNext();
        if(isItFirst)
        {
            this->head=multipliedCoin;
            isItFirst=false;
        }
    }*/
}


Blockchain& Blockchain::operator/=(int divisor)
{

    Koin *divideCoin=this->getHead();
    while(divideCoin!= nullptr)
    {
        divideCoin->KoinDivider(divisor);
        divideCoin=divideCoin->getNext();
    }
    /*
    Koin *divideCoin=this->getHead();
    Koin *dividedCoin;
    bool isItFirst=true;
    while(divideCoin!= nullptr)
    {
        //printf("!K");
        dividedCoin= new Koin(divideCoin->getValue()*divisor);
        dividedCoin->setNext(divideCoin->getNext());
        divideCoin=divideCoin->getNext();
        if(isItFirst)
        {
            this->head=dividedCoin;
            isItFirst=false;
        }
    }*/
}


std::ostream& operator<<(std::ostream& os, const Blockchain& blockchain)
{
    //USED FOR ERROR CHECKING printf("%0.3f XXX ",headkoin.getValue());
    int precision=Utilizer::koinPrintPrecision();
    os<<std::setprecision(precision);
    os<<std::fixed;
    os << "Block " << blockchain.getID()<<": ";
    Koin *osCoin=blockchain.getHead();
    if(osCoin== nullptr)
    {
        os<< "Empty.";
        return os;
    }
    else
    {
        while(osCoin!= nullptr)
        {
            //printf("!L");
            //USED FOR ERROR CHECKING printf("%0.3f XXX ",osKoin->getValue());
            os<<osCoin->getValue()<<"--";
            osCoin=osCoin->getNext();
        }
        os<<"|("<<blockchain.getTotalValue()<<")";
    }
    return os;
}

void Blockchain::changeForkType(std::string softness)
{
    if(softness=="soft") this->softFork=true;
    else this->softFork=false;
}

void Blockchain::setHead(Koin *newHead)
{
    if(this->head!= nullptr) return;
    else this->head=newHead;
}