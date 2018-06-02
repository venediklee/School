#include "Player.h"

/*
YOU MUST WRITE THE IMPLEMENTATIONS OF THE REQUESTED FUNCTIONS
IN THIS FILE. START YOUR IMPLEMENTATIONS BELOW THIS LINE
*/
#include "iomanip"
#include "sstream"
#include <iostream>
Player::Player(uint id2, int x, int y):id(id2),coordinate(x,y){};

//IMPLEMENT for each derived class
Player::~Player()=default;

uint Player::getID() const
{
    return this->id;
}
const Coordinate& Player::getCoord() const
{
    return (this->coordinate);
}

int Player::getHP() const
{
    return this->HP;
}
void Player::setHP(int newHP)
{
	this->HP=newHP;
}

std::string Player::getBoardID() const
{
    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << this->getID();
    return ss.str();
}

//IMPLEMENT for each derived class
//virtual Armor Player::getArmor() const;
//IMPLEMENT for each derived class
//virtual Weapon Player::getWeapon() const;


//IMPLEMENT for each derived class
/**
 * Every player has a different priority move list.
 * It's explained in the Players' header.
 *
 * @return The move priority list for the player.
 */
//virtual std::vector<Move> Player::getPriorityList() const = 0;

//IMPLEMENT for each derived class
/**
 * Get the full name of the player.
 *
 * Example (Tracer with ID 92) = "Tracer92"
 * Example (Tracer with ID 1)  = "Tracer01"
 *
 * @return Full name of the player.
 */
//virtual const std::string Player::getFullName() const = 0;


bool Player::isDead() const
{
    return this->getHP()<=0;
}



void Player::executeMove(Move move)
{
	this->coordinate=this->coordinate+(move);

	std::stringstream ss;
	ss<<this->getFullName()<<"("<<this->getHP()<<") moved ";
	switch(move)
    {
        case UP:ss<<"UP"; break;
        case DOWN:ss<<"DOWN";break;
        case RIGHT: ss<<"RIGHT";break;
        case LEFT:ss<<"LEFT";break;
        default:ss<<std::endl<<"YOU DONE FUCKED UP AT EXECUTEMOVE"<<std::endl;
    }
    ss<<"."<<std::endl;
	std::cout << ss.str();
}



bool Player::attackTo(Player *player)
{
    if(this==player) return false;
    Weapon lhsWeapon=(this->getWeapon());
    //use Entity::damageForWeapon()
    int damage=Entity::damageForWeapon(lhsWeapon);
    Armor rhsArmor=player->getArmor();
    //use Entity::damageReductionForArmor();
    int damageReduction=Entity::damageReductionForArmor(rhsArmor);

    //printing the attack
    std::stringstream ss;
    ss<<this->getFullName()<<"("<<this->getHP()<<
      ") ATTACKED "<<player->getFullName()<<
      "("<<player->getHP()<<")! (-"<<damage-damageReduction<<")"<<std::endl;
    std::cout<<ss.str();

    int newPlayerHP=player->getHP()-std::max((damage-damageReduction),0);
    player->setHP(newPlayerHP);
    return (player->getHP()>0)? false:true;
}

/*
bool Player::operator()(const Player* a, const Player* b)
{
    return a->getID()<b->getID();
}
*/
bool Player::operator<(const Player* const rhs) const
{
    return id<rhs->getID();
}
