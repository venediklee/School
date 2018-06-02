#ifndef HW4_PACIFIST_H
#define HW4_PACIFIST_H


#include "Player.h"

class Pacifist : public Player {
public:
    Pacifist(uint id, int x, int y);

    // Name     : Pacifist
    // Priority : { UP, LEFT, DOWN, RIGHT }

    // Armor    : METAL
    // Weapon   : NOWEAPON
    // HP       : 100

    // DO NOT MODIFY THE UPPER PART
    // ADD OWN PUBLIC METHODS/PROPERTIES/OVERRIDES BELOW

    ~Pacifist();
    Armor getArmor() const;
    Weapon getWeapon() const ;
    /**
     * Every player has a different priority move list.
     * It's explained in the Players' header.
     *
     * @return The move priority list for the player.
     */
    std::vector<Move> getPriorityList() const ;
    /**
     * Get the full name of the player.
     *
     * Example (Tracer with ID 92) = "Tracer92"
     * Example (Tracer with ID 1)  = "Tracer01"
     *
     * @return Full name of the player.
     */
    const std::string getFullName() const;
    //bool operator()(Pacifist a,Pacifist b);
    
};


#endif //HW4_PACIFIST_H
