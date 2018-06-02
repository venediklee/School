#include "Pacifist.h"

/*
YOU MUST WRITE THE IMPLEMENTATIONS OF THE REQUESTED FUNCTIONS
IN THIS FILE. START YOUR IMPLEMENTATIONS BELOW THIS LINE 
*/

#include "iomanip"
#include "sstream"
#include <iostream>

Pacifist::Pacifist(uint id, int x, int y):Player(id,x,y)
{
    Player::HP=100;

}

//IMPLEMENT check if you need to change anything
Pacifist::~Pacifist()=default;

Armor Pacifist::getArmor() const
{
    return METAL;
}
Weapon Pacifist::getWeapon() const
{
    return NOWEAPON;
}


std::vector<Move> Pacifist::getPriorityList() const
{
    return { UP, LEFT, DOWN, RIGHT };
}
/**
 * Get the full name of the player.
 *
 * Example (Tracer with ID 92) = "Tracer92"
 * Example (Tracer with ID 1)  = "Tracer01"
 *
 * @return Full name of the player.
 */
const std::string Pacifist::getFullName() const
{
    std::stringstream ss;
    ss<<"Pacifist";
    if(this->getID()<10) ss<<"0";
    ss<< this->getID();
    return ss.str();
}
/*
bool Pacifist::operator()(Pacifist a, Pacifist b)
{
    return a.getID()<b.getID();
}*/