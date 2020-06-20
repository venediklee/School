using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace cyclone
{
    using real = System.Double;

    /**
     * Links connect two particles together, generating a contact if
     * they violate the constraints of their link. It is used as a
     * base class for cables and rods, and could be used as a base
     * class for springs with a limit to their extension..
     */
    class ParticleLink :  ParticleContactGenerator
    {

        /**
         * Holds the pair of particles that are connected by this link.
         * there is probably an error by author about private-public with this var.
         */
        public Particle[] particle=new Particle[2];


        /**
         * Returns the current length of the link.
         */
        protected real currentLength()
        {
            MyVector3 relativePos = particle[0].GetPosition() -
                          particle[1].GetPosition();
            return relativePos.Magnitude();
        }


        /**
         * Geneates the contacts to keep this link from being
         * violated. This class can only ever generate a single
         * contact, so the pointer can be a pointer to a single
         * element, the limit parameter is assumed to be at least one
         * (zero isn't valid) and the return value is either 0, if the
         * cable wasn't over-extended, or one if a contact was needed.
         *
         * NB: This method is declared in the same way (as pure
         * virtual) in the parent class, but is replicated here for
         * documentation purposes.
         */
        public virtual uint addContact(ParticleContact contact,
                                    uint limit)
        {
            return 0;
        }
    };


    /**
     * Cables link a pair of particles, generating a contact if they
     * stray too far apart.
     */
    class ParticleCable : ParticleLink
    {
    
        /**
         * Holds the maximum length of the cable.
         */
        public real maxLength;

        /**
         * Holds the restitution (bounciness) of the cable.
         */
        public real restitution;


        /**
         * Fills the given contact structure with the contact needed
         * to keep the cable from over-extending.
         */
        public virtual uint addContact(ParticleContact contact, uint limit)
        {
            // Find the length of the cable
            real length = currentLength();

            // Check if we're over-extended
            if (length < maxLength)
            {
                return 0;
            }

            // Otherwise return the contact
            contact.particle[0] = particle[0];
            contact.particle[1] = particle[1];

            // Calculate the normal
            MyVector3 normal = particle[1].GetPosition() - particle[0].GetPosition();
            normal.Normalise();
            contact.contactNormal = normal;

            contact.penetration = length - maxLength;
            contact.restitution = restitution;

            return 1;
        }
    };


    /**
     * Rods link a pair of particles, generating a contact if they
     * stray too far apart or too close.
     */
    class ParticleRod : ParticleLink
    {

        /**
         * Holds the length of the rod.
         */
        public real length;

        /**
         * Fills the given contact structure with the contact needed
         * to keep the rod from extending or compressing.
         */
       public virtual uint addContact(ParticleContact contact,
                                     uint limit)
        {
            // Find the length of the rod
            real currentLen = currentLength();

            // Check if we're over-extended
            if (currentLen == length)
            {
                return 0;
            }

            // Otherwise return the contact
            contact.particle[0] = particle[0];
            contact.particle[1] = particle[1];

            // Calculate the normal
            MyVector3 normal = particle[1].GetPosition() - particle[0].GetPosition();
            normal.Normalise();

            // The contact normal depends on whether we're extending or compressing
            if (currentLen > length)
            {
                contact.contactNormal = normal;
                contact.penetration = currentLen - length;
            }
            else
            {
                contact.contactNormal = normal * -1;
                contact.penetration = length - currentLen;
            }

            // Always use zero restitution (no bounciness)
            contact.restitution = 0;

            return 1;
        }
    };

    /**
    * Constraints are just like links, except they connect a particle to
    * an immovable anchor point.
    */
    class ParticleConstraint : ParticleContactGenerator
    {

        /**
        * Holds the particles connected by this constraint.
        */
        public Particle particle;

        /**
         * The point to which the particle is anchored.
         */
        public MyVector3 anchor;


        /**
        * Returns the current length of the link.
        */
        protected real currentLength()
        {
            MyVector3 relativePos = particle.GetPosition() - anchor;
            return relativePos.Magnitude();
        }

    
        /**
        * Geneates the contacts to keep this link from being
        * violated. This class can only ever generate a single
        * contact, so the pointer can be a pointer to a single
        * element, the limit parameter is assumed to be at least one
        * (zero isn't valid) and the return value is either 0, if the
        * cable wasn't over-extended, or one if a contact was needed.
        *
        * NB: This method is declared in the same way (as pure
        * virtual) in the parent class, but is replicated here for
        * documentation purposes.
        */
       public virtual uint addContact(ParticleContact contact,
            uint limit)
        {
            return 0;
        }
    };

    /**
    * Cables link a particle to an anchor point, generating a contact if they
    * stray too far apart.
    */
    class ParticleCableConstraint : ParticleConstraint
    {
    
        /**
        * Holds the maximum length of the cable.
        */
        public real maxLength;

        /**
        * Holds the restitution (bounciness) of the cable.
        */
        public real restitution;
    
        /**
        * Fills the given contact structure with the contact needed
        * to keep the cable from over-extending.
        */
        public virtual uint addContact(ParticleContact contact, uint limit)
        {
            // Find the length of the cable
            real length = currentLength();

            // Check if we're over-extended
            if (length < maxLength)
            {
                return 0;
            }

            // Otherwise return the contact
            contact.particle[0] = particle;
            contact.particle[1] = null;

            // Calculate the normal
            MyVector3 normal = anchor - particle.GetPosition();
            normal.Normalise();
            contact.contactNormal = normal;

            contact.penetration = length - maxLength;
            contact.restitution = restitution;

            return 1;
        }
    };

    /**
    * Rods link a particle to an anchor point, generating a contact if they
    * stray too far apart or too close.
    */
    class ParticleRodConstraint : ParticleConstraint
    {
    
        /**
        * Holds the length of the rod.
        */
        public real length;
    
        /**
        * Fills the given contact structure with the contact needed
        * to keep the rod from extending or compressing.
        */
        public virtual uint addContact(ParticleContact contact, uint limit)
        {
            // Find the length of the rod
            real currentLen = currentLength();

            // Check if we're over-extended
            if (currentLen == length)
            {
                return 0;
            }

            // Otherwise return the contact
            contact.particle[0] = particle;
            contact.particle[1] = null;

            // Calculate the normal
            MyVector3 normal = anchor - particle.GetPosition();
            normal.Normalise();

            // The contact normal depends on whether we're extending or compressing
            if (currentLen > length)
            {
                contact.contactNormal = normal;
                contact.penetration = currentLen - length;
            }
            else
            {
                contact.contactNormal = normal * -1;
                contact.penetration = length - currentLen;
            }

            // Always use zero restitution (no bounciness)
            contact.restitution = 0;

            return 1;
        }
    };

}
