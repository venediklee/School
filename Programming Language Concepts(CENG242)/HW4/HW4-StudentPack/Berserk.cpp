#include "Berserk.h"

/*
YOU MUST WRITE THE IMPLEMENTATIONS OF THE REQUESTED FUNCTIONS
IN THIS FILE. START YOUR IMPLEMENTATIONS BELOW THIS LINE
*/
#include "iomanip"
#include "sstream"
#include <iostream>

Berserk::Berserk(uint id, int x, int y):Player(id,x,y)
{
  Player::HP=100;

}

//IMPLEMENT check if you need to change anything
Berserk::~Berserk()=default;

Armor Berserk::getArmor() const
{
  return WOODEN;
}
Weapon Berserk::getWeapon() const
{
  return PISTOL;
}


std::vector<Move> Berserk::getPriorityList() const
{
  return { ATTACK, UP, LEFT, DOWN, RIGHT };
}
/**
 * Get the full name of the player.
 *
 * Example (Tracer with ID 92) = "Tracer92"
 * Example (Tracer with ID 1)  = "Tracer01"
 *
 * @return Full name of the player.
 */
const std::string Berserk::getFullName() const
{
  std::stringstream ss;
	ss<<"Berserk";
	if(this->getID()<10) ss<<"0";
	ss<< this->getID();
	return ss.str();
}
/*
bool Berserk::operator()(Berserk a, Berserk b)
{
    return a.getID()<b.getID();
}*/