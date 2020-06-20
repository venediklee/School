using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace cyclone
{
    using real = System.Double;//precision

    /**
     * A Contact represents two objects in contact (in this case
     * ParticleContact representing two Particles). Resolving a
     * contact removes their interpenetration, and applies sufficient
     * impulse to keep them apart. Colliding bodies may also rebound.
     *
     * The contact has no callable functions, it just holds the
     * contact details. To resolve a set of contacts, use the particle
     * contact resolver class.
     */
    class ParticleContact
    {
        // ... Other ParticleContact code as before ...

        /**
        * The contact resolver object needs access into the contacts to
        * set and effect the contact.
        */
        //friend class ParticleContactResolver;



        /**
         * Holds the particles that are involved in the contact. The
         * second of these can be NULL, for contacts with the scenery.
         */
        public Particle [] particle= new Particle[2];

        /**
         * Holds the normal restitution coefficient at the contact.
         */
        public real restitution;

        /**
         * Holds the direction of the contact in world coordinates.
         */
        public MyVector3 contactNormal;

        /**
         * Holds the depth of penetration at the contact.
         */
        public real penetration;

        /**
         * Holds the amount each particle is moved by during interpenetration
         * resolution.
         */
        public MyVector3[] particleMovement = new MyVector3[2];


        /**
         * Resolves this contact, for both velocity and interpenetration.
         */
        //TODO this is supposed to be protected with contact resolver class as a friend of this class
        public void resolve(real duration)
        {
            resolveVelocity(duration);
            resolveInterpenetration(duration);
        }

        /**
         * Calculates the separating velocity at this contact.
         */
        //TODO this is supposed to be protected with contact resolver class as a friend of this class
        public real calculateSeparatingVelocity()
        {
            MyVector3 relativeVelocity = particle[0].GetVelocity();
            if (particle[1]!=null) relativeVelocity -= particle[1].GetVelocity();
            return relativeVelocity * contactNormal;
        }


        /**
         * Handles the impulse calculations for this collision.
         */
        private void resolveVelocity(real duration)
        {
            // Find the velocity in the direction of the contact
            real separatingVelocity = calculateSeparatingVelocity();

            // Check if it needs to be resolved
            if (separatingVelocity > 0)
            {
                // The contact is either separating, or stationary - there's
                // no impulse required.
                return;
            }

            // Calculate the new separating velocity
            real newSepVelocity = -separatingVelocity * restitution;

            // Check the velocity build-up due to acceleration only
            MyVector3 accCausedVelocity = particle[0].GetAcceleration();
            if (particle[1]!=null) accCausedVelocity -= particle[1].GetAcceleration();
            real accCausedSepVelocity = accCausedVelocity * contactNormal * duration;

            // If we've got a closing velocity due to acceleration build-up,
            // remove it from the new separating velocity
            if (accCausedSepVelocity < 0)
            {
                newSepVelocity += restitution * accCausedSepVelocity;

                // Make sure we haven't removed more than was
                // there to remove.
                if (newSepVelocity < 0) newSepVelocity = 0;
            }

            real deltaVelocity = newSepVelocity - separatingVelocity;

            // We apply the change in velocity to each object in proportion to
            // their inverse mass (i.e. those with lower inverse mass [higher
            // actual mass] get less change in velocity)..
            real totalInverseMass = particle[0].GetInverseMass();
            if (particle[1]!=null) totalInverseMass += particle[1].GetInverseMass();

            // If all particles have infinite mass, then impulses have no effect
            if (totalInverseMass <= 0) return;

            // Calculate the impulse to apply
            real impulse = deltaVelocity / totalInverseMass;

            // Find the amount of impulse per unit of inverse mass
            MyVector3 impulsePerIMass = contactNormal * impulse;

            // Apply impulses: they are applied in the direction of the contact,
            // and are proportional to the inverse mass.
            particle[0].SetVelocity(particle[0].GetVelocity() +
                impulsePerIMass * particle[0].GetInverseMass()
                );
            if (particle[1]!=null)
            {
                // Particle 1 goes in the opposite direction
                particle[1].SetVelocity(particle[1].GetVelocity() +
                    impulsePerIMass * -particle[1].GetInverseMass()
                    );
            }
        }

        /**
         * Handles the interpenetration resolution for this contact.
         */
        private void resolveInterpenetration(real duration)
        {
            // If we don't have any penetration, skip this step.
            if (penetration <= 0) return;

            // The movement of each object is based on their inverse mass, so
            // total that.
            real totalInverseMass = particle[0].GetInverseMass();
            if (particle[1]!=null) totalInverseMass += particle[1].GetInverseMass();

            // If all particles have infinite mass, then we do nothing
            if (totalInverseMass <= 0) return;

            // Find the amount of penetration resolution per unit of inverse mass
            MyVector3 movePerIMass = contactNormal * (penetration / totalInverseMass);

            // Calculate the the movement amounts
            particleMovement[0] = movePerIMass * particle[0].GetInverseMass();
            if (particle[1]!=null)
            {
                particleMovement[1] = movePerIMass * -particle[1].GetInverseMass();
            }
            else
            {
                particleMovement[1].Clear();
            }

            // Apply the penetration resolution
            particle[0].SetPosition(particle[0].GetPosition() + particleMovement[0]);
            if (particle[1]!=null)
            {
                particle[1].SetPosition(particle[1].GetPosition() + particleMovement[1]);
            }
        }
    };


    /**
     * The contact resolution routine for particle contacts. One
     * resolver instance can be shared for the whole simulation.
     */
    class ParticleContactResolver
    {

        /**
         * Holds the number of iterations allowed.
         */
        protected uint iterations;

        /**
         * This is a performance tracking value - we keep a record
         * of the actual number of iterations used.
         */
        protected uint iterationsUsed;



        /**
         * Creates a new contact resolver.
         */
        public ParticleContactResolver(uint iterations)
        {
            this.iterations = iterations;
        }

        /**
         * Sets the number of iterations that can be used.
         */
        public void setIterations(uint iterations)
        {
            this.iterations = iterations;
        }

        /**
         * Resolves a set of particle contacts for both penetration
         * and velocity.
         *
         * Contacts that cannot interact with each other should be
         * passed to separate calls to resolveContacts, as the
         * resolution algorithm takes much longer for lots of contacts
         * than it does for the same number of contacts in small sets.
         *
         * @param contactArray Pointer to an array of particle contact
         * objects.
         *
         * @param numContacts The number of contacts in the array to
         * resolve.
         *
         * @param numIterations The number of iterations through the
         * resolution algorithm. This should be at least the number of
         * contacts (otherwise some constraints will not be resolved -
         * although sometimes this is not noticable). If the
         * iterations are not needed they will not be used, so adding
         * more iterations may not make any difference. But in some
         * cases you would need millions of iterations. Think about
         * the number of iterations as a bound: if you specify a large
         * number, sometimes the algorithm WILL use it, and you may
         * drop frames.
         *
         * @param duration The duration of the previous integration step.
         * This is used to compensate for forces applied.
        */
        public void resolveContacts(ParticleContact []contactArray, uint numContacts, real duration)
        {
            uint i;

            iterationsUsed = 0;
            while (iterationsUsed < iterations)
            {
                // Find the contact with the largest closing velocity;
                real max = real.MaxValue;
                uint maxIndex = numContacts;
                for (i = 0; i < numContacts; i++)
                {
                    real sepVel = contactArray[i].calculateSeparatingVelocity();
                    if (sepVel < max &&
                        (sepVel < 0 || contactArray[i].penetration > 0))
                    {
                        max = sepVel;
                        maxIndex = i;
                    }
                }

                // Do we have anything worth resolving?
                if (maxIndex == numContacts) break;

                // Resolve this contact
                contactArray[maxIndex].resolve(duration);

                // Update the interpenetrations for all particles
                MyVector3[] move = contactArray[maxIndex].particleMovement;
                for (i = 0; i < numContacts; i++)
                {
                    if (contactArray[i].particle[0] == contactArray[maxIndex].particle[0])
                    {
                        contactArray[i].penetration -= move[0] * contactArray[i].contactNormal;
                    }
                    else if (contactArray[i].particle[0] == contactArray[maxIndex].particle[1])
                    {
                        contactArray[i].penetration -= move[1] * contactArray[i].contactNormal;
                    }
                    if (contactArray[i].particle[1]!=null)
                    {
                        if (contactArray[i].particle[1] == contactArray[maxIndex].particle[0])
                        {
                            contactArray[i].penetration += move[0] * contactArray[i].contactNormal;
                        }
                        else if (contactArray[i].particle[1] == contactArray[maxIndex].particle[1])
                        {
                            contactArray[i].penetration += move[1] * contactArray[i].contactNormal;
                        }
                    }
                }

                iterationsUsed++;
            }
        }
    };

    /**
     * This is the basic polymorphic interface for contact generators
     * applying to particles.
     */
    class ParticleContactGenerator
    {

        /**
         * Fills the given contact structure with the generated
         * contact. The contact pointer should point to the first
         * available contact in a contact array, where limit is the
         * maximum number of contacts in the array that can be written
         * to. The method returns the number of contacts that have
         * been written.
         */
        public virtual uint addContact(ParticleContact[] contact, uint limit)
        {
            return 0;
        }
    };

}
