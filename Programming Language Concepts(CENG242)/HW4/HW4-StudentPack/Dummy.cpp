#include "Dummy.h"

/*
YOU MUST WRITE THE IMPLEMENTATIONS OF THE REQUESTED FUNCTIONS
IN THIS FILE. START YOUR IMPLEMENTATIONS BELOW THIS LINE 
*/

#include "iomanip"
#include "sstream"
#include <iostream>

Dummy::Dummy(uint id, int x, int y):Player(id,x,y)
{
    Player::HP=1000;

}

//IMPLEMENT check if you need to change anything
Dummy::~Dummy()=default;

Armor Dummy::getArmor() const
{
    return NOARMOR;
}
Weapon Dummy::getWeapon() const
{
    return NOWEAPON;
}


std::vector<Move> Dummy::getPriorityList() const
{
    return { NOOP };
}
/**
 * Get the full name of the player.
 *
 * Example (Tracer with ID 92) = "Tracer92"
 * Example (Tracer with ID 1)  = "Tracer01"
 *
 * @return Full name of the player.
 */
const std::string Dummy::getFullName() const
{
    std::stringstream ss;
    ss<<"Dummy";
    if(this->getID()<10) ss<<"0";
    ss<< this->getID();
    return ss.str();
}
/*
bool Dummy::operator()(Dummy a, Dummy b)
{
    return a.getID()<b.getID();
}*/