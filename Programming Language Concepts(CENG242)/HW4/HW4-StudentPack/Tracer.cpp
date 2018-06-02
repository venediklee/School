#include "Tracer.h"

/*
YOU MUST WRITE THE IMPLEMENTATIONS OF THE REQUESTED FUNCTIONS
IN THIS FILE. START YOUR IMPLEMENTATIONS BELOW THIS LINE 
*/

#include "iomanip"
#include "sstream"
#include <iostream>
Tracer::Tracer(uint id, int x, int y):Player(id,x,y)
{
    Player::HP=100;

}

//IMPLEMENT check if you need to change anything
Tracer::~Tracer()=default;

Armor Tracer::getArmor() const
{
    return BRICK;
}
Weapon Tracer::getWeapon() const
{
    return SHOVEL;
}


std::vector<Move> Tracer::getPriorityList() const
{
    return { UP, LEFT, DOWN, RIGHT, ATTACK };
}
/**
 * Get the full name of the player.
 *
 * Example (Tracer with ID 92) = "Tracer92"
 * Example (Tracer with ID 1)  = "Tracer01"
 *
 * @return Full name of the player.
 */
const std::string Tracer::getFullName() const
{
    std::stringstream ss;
    ss<<"Tracer";
    if(this->getID()<10) ss<<"0";
    ss<< this->getID();
    return ss.str();
}

/*
bool Tracer::operator()(Tracer a, Tracer b)
{
    return a.getID()<b.getID();
}*/