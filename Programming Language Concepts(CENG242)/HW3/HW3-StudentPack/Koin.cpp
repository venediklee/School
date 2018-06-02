#include <iostream>
#include "Koin.h"

/*
YOU MUST WRITE THE IMPLEMENTATIONS OF THE REQUESTED FUNCTIONS
IN THIS FILE. START YOUR IMPLEMENTATIONS BELOW THIS LINE 
*/
#include "Utilizer.h"
#include "iomanip"

Koin::Koin():value(),next(nullptr){}//default constructor

Koin::Koin(double value): value(value),next() {}//constructor

Koin::Koin(const Koin& rhs) //copy constructor
{
    //*this=NULL;
    if(&(rhs)!=nullptr) *this=rhs;
}

Koin& Koin::operator=(const Koin& rhs)// assignment(=) operator
{
    //*this=NULL;

    if(&(rhs)!=nullptr)
    {
        this->value=rhs.value;
        this->next=rhs.next;
    }
    return *this;
}

Koin::~Koin()
{
    //this->value=NULL;
    this->next=nullptr;
}

double Koin::getValue() const
{
    if(this!=nullptr)
    {
        return this->value;
    }
    //return NULL;
}

Koin* Koin::getNext() const
{
    if(this!=nullptr)
    {
        return this->next;
    }
    else return nullptr;
}

void Koin::setNext(Koin *next)
{
    if(this!= nullptr)
    {
        this->next=next;
    }
    else this->next=nullptr;
}

bool Koin::operator==(const Koin& rhs) const
{
    double absDiff=((this->getValue()-rhs.getValue()))? (this->getValue()-rhs.getValue()): -(this->getValue()-rhs.getValue());
    if(absDiff<Utilizer::doubleSensitivity()) return (this->getNext())==rhs.getNext();
    else return false;
}

bool Koin::operator!=(const Koin& rhs) const
{
    return !(*this==rhs);
}

Koin& Koin::operator*=(int multiplier)
{
    this->value=this->getValue()*((double)multiplier);
    return *this;
}

Koin& Koin::operator/=(int divisor)
{
    this->value=this->getValue()/((double)divisor);
    return *this;
}

std::ostream& operator<<(std::ostream& os, const Koin& koin)
{
    int precision=Utilizer::koinPrintPrecision();
    if(&koin!= nullptr)
    {
        //USED FOR ERROR CHECKING printf("%0.3f XXX ",koin.getValue());
        os<<std::setprecision(precision)<<std::fixed << (koin.getValue());
        Koin *osKoin=koin.getNext();//for the while loop
        while(osKoin!= nullptr)
        {
            //USED FOR ERROR CHECKING printf("%0.3f XXX ",osKoin->getValue());
            os<<"--"<<osKoin->getValue();
            osKoin=osKoin->getNext();
        }
        os<<"--|";
    }

    return os;
}
