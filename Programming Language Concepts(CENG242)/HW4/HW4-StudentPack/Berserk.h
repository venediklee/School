#ifndef HW4_BERSERK_H
#define HW4_BERSERK_H


#include "Player.h"

class Berserk : public Player {
public:
    Berserk(uint id, int x, int y);

    // Name     : Berserk
    // Priority : { ATTACK, UP, LEFT, DOWN, RIGHT }

    // Armor    : WOODEN
    // Weapon   : PISTOL
    // HP       : 100

    // DO NOT MODIFY THE UPPER PART
    // ADD OWN PUBLIC METHODS/PROPERTIES/OVERRIDES BELOW



    ~Berserk();
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
    //bool operator()(Berserk a,Berserk b);
};


#endif //HW4_BERSERK_H
