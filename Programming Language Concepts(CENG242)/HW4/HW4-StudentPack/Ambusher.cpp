#include "Ambusher.h"

/*
YOU MUST WRITE THE IMPLEMENTATIONS OF THE REQUESTED FUNCTIONS
IN THIS FILE. START YOUR IMPLEMENTATIONS BELOW THIS LINE 
*/
#include "iomanip"
#include "sstream"
#include <iostream>

Ambusher::Ambusher(uint id, int x, int y):Player(id,x,y)
{
    Player::HP=100;

}

//IMPLEMENT check if you need to change anything
Ambusher::~Ambusher()=default;

Armor Ambusher::getArmor() const
{
    return NOARMOR;
}
Weapon Ambusher::getWeapon() const
{
    return SEMIAUTO;
}


std::vector<Move> Ambusher::getPriorityList() const
{
    return { ATTACK };
}
/**
 * Get the full name of the player.
 *
 * Example (Tracer with ID 92) = "Tracer92"
 * Example (Tracer with ID 1)  = "Tracer01"
 *
 * @return Full name of the player.
 */
const std::string Ambusher::getFullName() const
{
    std::stringstream ss;
    ss<<"Ambusher";
    if(this->getID()<10) ss<<"0";
    ss<< this->getID();
    return ss.str();
}
/*bool Ambusher::operator()(Ambusher a, Ambusher b)
{
    return a.getID()<b.getID();
}*/
