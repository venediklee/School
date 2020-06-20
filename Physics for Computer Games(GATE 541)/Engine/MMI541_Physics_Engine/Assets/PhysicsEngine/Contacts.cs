using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System;
namespace cyclone
{
    using real = System.Double;


    /**
     * A contact represents two bodies in contact. Resolving a
     * contact removes their interpenetration, and applies sufficient
     * impulse to keep them apart. Colliding bodies may also rebound.
     * Contacts can be used to represent positional joints, by making
     * the contact constraint keep the bodies in their correct
     * orientation.
     *
     * It can be a good idea to create a contact object even when the
     * contact isn't violated. Because resolving one contact can violate
     * another, contacts that are close to being violated should be
     * sent to the resolver; that way if one resolution moves the body,
     * the contact may be violated, and can be resolved. If the contact
     * is not violated, it will not be resolved, so you only loose a
     * small amount of execution time.
     *
     * The contact has no callable functions, it just holds the contact
     * details. To resolve a set of contacts, use the contact resolver
     * class.
     */
    class Contact
    {
        // ... Other data as before ...

        /**
         * The contact resolver object needs access into the contacts to
         * set and effect the contact.
         */
        //friend class ContactResolver;


        /**
         * Holds the bodies that are involved in the contact. The
         * second of these can be NULL, for contacts with the scenery.
         */
        public RigidBody[] body = new RigidBody[2];

        /**
         * Holds the lateral friction coefficient at the contact.
         */
        public real friction;

        /**
         * Holds the normal restitution coefficient at the contact.
         */
        public real restitution;

        /**
         * Holds the position of the contact in world coordinates.
         */
        public MyVector3 contactPoint;

        /**
         * Holds the direction of the contact in world coordinates.
         */
        public MyVector3 contactNormal;

        /**
         * Holds the depth of penetration at the contact point. If both
         * bodies are specified then the contact point should be midway
         * between the inter-penetrating points.
         */
        public real penetration;

        /**
         * Sets the data that doesn't normally depend on the position
         * of the contact (i.e. the bodies, and their material properties).
         */
        public void setBodyData(RigidBody one, RigidBody two,
                         real friction, real restitution)
        {
            body[0] = one;
            body[1] = two;
            this.friction = friction;
            this.restitution = restitution;
        }


        //DEV_NOTE this is normally protected
        /**
         * A transform matrix that converts co-ordinates in the contact's
         * frame of reference to world co-ordinates. The columns of this
         * matrix form an orthonormal set of vectors.
         */
        public Matrix3 contactToWorld;

        //DEV_NOTE this is normally protected
        /**
         * Holds the closing velocity at the point of contact. This is set
         * when the calculateInternals function is run.
         */
        public MyVector3 contactVelocity;

        //DEV_NOTE this is normally protected
        /**
         * Holds the required change in velocity for this contact to be
         * resolved.
         */
        public real desiredDeltaVelocity;

        //DEV_NOTE this is normally protected
        /**
         * Holds the world space position of the contact point relative to
         * centre of each body. This is set when the calculateInternals
         * function is run.
         */
        public MyVector3[] relativeContactPosition = new MyVector3[2];

        //DEV_NOTE this is normally protected
        /**
         * Calculates internal data from state data. This is called before
         * the resolution algorithm tries to do any resolution. It should
         * never need to be called manually.
         */
        public void calculateInternals(real duration)
        {
            // Check if the first object is NULL, and swap if it is.
            if (body[0] == null) swapBodies();
            Debug.Assert(body[0] != null);

            // Calculate an set of axis at the contact point.
            calculateContactBasis();

            // Store the relative position of the contact relative to each body
            relativeContactPosition[0] = contactPoint - body[0].getPosition();
            if (body[1] != null)
            {
                relativeContactPosition[1] = contactPoint - body[1].getPosition();
            }

            // Find the relative velocity of the bodies at the contact point.
            contactVelocity = calculateLocalVelocity(0, duration);
            if (body[1] != null)
            {
                contactVelocity -= calculateLocalVelocity(1, duration);
            }

            // Calculate the desired change in velocity for resolution
            calculateDesiredDeltaVelocity(duration);
        }

        //DEV_NOTE this is normally protected
        /**
         * Reverses the contact. This involves swapping the two rigid bodies
         * and reversing the contact normal. The internal values should then
         * be recalculated using calculateInternals (this is not done
         * automatically).
         */
        public void swapBodies()
        {
            contactNormal *= -1;

            RigidBody temp = body[0];
            body[0] = body[1];
            body[1] = temp;
        }

        //DEV_NOTE this is normally protected
        /**
         * Updates the awake state of rigid bodies that are taking
         * place in the given contact. A body will be made awake if it
         * is in contact with a body that is awake.
         */
        public void matchAwakeState()
        {
            // Collisions with the world never cause a body to wake up.
            if (body[1] == null) return;

            bool body0awake = body[0].getAwake();
            bool body1awake = body[1].getAwake();

            // Wake up only the sleeping one
            if (body0awake ^ body1awake)
            {
                if (body0awake) body[1].setAwake();
                else body[0].setAwake();
            }
        }

        //DEV_NOTE this is normally protected
        /**
         * Calculates and sets the internal value for the desired delta
         * velocity.
         */
        public void calculateDesiredDeltaVelocity(real duration)
        {
            real velocityLimit = (real)0.25f;

            // Calculate the acceleration induced velocity accumulated this frame
            real velocityFromAcc = 0;

            if (body[0].getAwake())
            {
                velocityFromAcc +=
                    body[0].getLastFrameAcceleration() * duration * contactNormal;
            }

            if (body[1] != null && body[1].getAwake())
            {
                velocityFromAcc -=
                    body[1].getLastFrameAcceleration() * duration * contactNormal;
            }

            // If the velocity is very slow, limit the restitution
            real thisRestitution = restitution;
            if (Math.Abs(contactVelocity.x) < velocityLimit)
            {
                thisRestitution = (real)0.0f;
            }

            // Combine the bounce velocity with the removed
            // acceleration velocity.
            desiredDeltaVelocity =
                -contactVelocity.x
                - thisRestitution * (contactVelocity.x - velocityFromAcc);
        }

        //DEV_NOTE this is normally protected
        /**
         * Calculates and returns the velocity of the contact
         * point on the given body.
         */
        public MyVector3 calculateLocalVelocity(uint bodyIndex, real duration)
        {
            RigidBody thisBody = body[bodyIndex];

            // Work out the velocity of the contact point.
            MyVector3 velocity =
                thisBody.getRotation() % relativeContactPosition[bodyIndex];
            velocity += thisBody.getVelocity();

            // Turn the velocity into contact-coordinates.
            MyVector3 contactVelocity = contactToWorld.transformTranspose(velocity);

            // Calculate the ammount of velocity that is due to forces without
            // reactions.
            MyVector3 accVelocity = thisBody.getLastFrameAcceleration() * duration;

            // Calculate the velocity in contact-coordinates.
            accVelocity = contactToWorld.transformTranspose(accVelocity);

            // We ignore any component of acceleration in the contact normal
            // direction, we are only interested in planar acceleration
            accVelocity.x = 0;

            // Add the planar velocities - if there's enough friction they will
            // be removed during velocity resolution
            contactVelocity += accVelocity;

            // And return it
            return contactVelocity;
        }

        //DEV_NOTE this is normally protected
        /**
         * Calculates an orthonormal basis for the contact point, based on
         * the primary friction direction (for anisotropic friction) or
         * a random orientation (for isotropic friction).
         */
        public void calculateContactBasis()
        {
            MyVector3[] contactTangent = new MyVector3[2];

            // Check whether the Z-axis is nearer to the X or Y axis
            if (Math.Abs(contactNormal.x) > Math.Abs(contactNormal.y))
            {
                // Scaling factor to ensure the results are normalised
                real s = (real)1.0f / Math.Sqrt(contactNormal.z * contactNormal.z +
                    contactNormal.x * contactNormal.x);

                // The new X-axis is at right angles to the world Y-axis
                contactTangent[0].x = contactNormal.z * s;
                contactTangent[0].y = 0;
                contactTangent[0].z = -contactNormal.x * s;

                // The new Y-axis is at right angles to the new X- and Z- axes
                contactTangent[1].x = contactNormal.y * contactTangent[0].x;
                contactTangent[1].y = contactNormal.z * contactTangent[0].x -
                    contactNormal.x * contactTangent[0].z;
                contactTangent[1].z = -contactNormal.y * contactTangent[0].x;
            }
            else
            {
                // Scaling factor to ensure the results are normalised
                real s = (real)1.0 / Math.Sqrt(contactNormal.z * contactNormal.z +
                    contactNormal.y * contactNormal.y);

                // The new X-axis is at right angles to the world X-axis
                contactTangent[0].x = 0;
                contactTangent[0].y = -contactNormal.z * s;
                contactTangent[0].z = contactNormal.y * s;

                // The new Y-axis is at right angles to the new X- and Z- axes
                contactTangent[1].x = contactNormal.y * contactTangent[0].z -
                    contactNormal.z * contactTangent[0].y;
                contactTangent[1].y = -contactNormal.x * contactTangent[0].z;
                contactTangent[1].z = contactNormal.x * contactTangent[0].y;
            }

            // Make a matrix from the three vectors.
            contactToWorld.setComponents(
                contactNormal,
                contactTangent[0],
                contactTangent[1]);
        }

        //DEV_NOTE not implemented in cyclone
        /**
         * Applies an impulse to the given body, returning the
         * change in velocities.
         */
        //protected void applyImpulse(MyVector3 impulse, RigidBody body,
        //                  MyVector3 velocityChange, MyVector3 rotationChange);

        //DEV_NOTE not implemented in cyclone
        /**
         * Performs an inertia-weighted impulse based resolution of this
         * contact alone.
         */
        public void applyVelocityChange(MyVector3[] velocityChange,
                                 MyVector3[] rotationChange)
        {
            // Get hold of the inverse mass and inverse inertia tensor, both in
            // world coordinates.
            Matrix3[] inverseInertiaTensor = new Matrix3[2];
            body[0].getInverseInertiaTensorWorld(inverseInertiaTensor[0]);
            if (body[1] != null)
                body[1].getInverseInertiaTensorWorld(inverseInertiaTensor[1]);

            // We will calculate the impulse for each contact axis
            MyVector3 impulseContact;

            if (friction == (real)0.0)
            {
                // Use the short format for frictionless contacts
                impulseContact = calculateFrictionlessImpulse(inverseInertiaTensor);
            }
            else
            {
                // Otherwise we may have impulses that aren't in the direction of the
                // contact, so we need the more complex version.
                impulseContact = calculateFrictionImpulse(inverseInertiaTensor);
            }

            // Convert impulse to world coordinates
            MyVector3 impulse = contactToWorld.transform(impulseContact);

            // Split in the impulse into linear and rotational components
            MyVector3 impulsiveTorque = relativeContactPosition[0] % impulse;
            rotationChange[0] = inverseInertiaTensor[0].transform(impulsiveTorque);
            velocityChange[0].Clear();
            velocityChange[0].AddScaledVector(impulse, body[0].getInverseMass());

            // Apply the changes
            body[0].addVelocity(velocityChange[0]);
            body[0].addRotation(rotationChange[0]);

            if (body[1] != null)
            {
                // Work out body one's linear and angular changes
                impulsiveTorque = new MyVector3(impulse % relativeContactPosition[1]);
                rotationChange[1] = inverseInertiaTensor[1].transform(impulsiveTorque);
                velocityChange[1].Clear();
                velocityChange[1].AddScaledVector(impulse, -body[1].getInverseMass());

                // And apply them.
                body[1].addVelocity(velocityChange[1]);
                body[1].addRotation(rotationChange[1]);
            }
        }

        //DEV_NOTE not implemented in cyclone
        /**
         * Performs an inertia weighted penetration resolution of this
         * contact alone.
         */
        public void applyPositionChange(MyVector3[] linearChange,
                                 MyVector3[] angularChange,
                                 real penetration)
        {
            const real angularLimit = (real)0.2f;
            real[] angularMove = new real[2];
            real[] linearMove = new real[2];

            real totalInertia = 0;
            real[] linearInertia = new real[2];
            real[] angularInertia = new real[2];

            // We need to work out the inertia of each object in the direction
            // of the contact normal, due to angular inertia only.
            for (uint i = 0; i < 2; i++) if (body[i] != null)
                {
                    Matrix3 inverseInertiaTensor = new Matrix3();
                    body[i].getInverseInertiaTensorWorld(inverseInertiaTensor);

                    // Use the same procedure as for calculating frictionless
                    // velocity change to work out the angular inertia.
                    MyVector3 angularInertiaWorld =
                        relativeContactPosition[i] % contactNormal;
                    angularInertiaWorld =
                        inverseInertiaTensor.transform(angularInertiaWorld);
                    angularInertiaWorld =
                        angularInertiaWorld % relativeContactPosition[i];
                    angularInertia[i] =
                        angularInertiaWorld * contactNormal;

                    // The linear component is simply the inverse mass
                    linearInertia[i] = body[i].getInverseMass();

                    // Keep track of the total inertia from all components
                    totalInertia += linearInertia[i] + angularInertia[i];

                    // We break the loop here so that the totalInertia value is
                    // completely calculated (by both iterations) before
                    // continuing.
                }

            // Loop through again calculating and applying the changes
            for (uint i = 0; i < 2; i++) if (body[i] != null)
                {
                    // The linear and angular movements required are in proportion to
                    // the two inverse inertias.
                    real sign = (i == 0) ? 1 : -1;
                    angularMove[i] =
                        sign * penetration * (angularInertia[i] / totalInertia);
                    linearMove[i] =
                        sign * penetration * (linearInertia[i] / totalInertia);

                    // To avoid angular projections that are too great (when mass is large
                    // but inertia tensor is small) limit the angular move.
                    MyVector3 projection = relativeContactPosition[i];
                    projection.AddScaledVector(
                        contactNormal,
                        -relativeContactPosition[i].ScalarProduct(contactNormal)
                        );

                    // Use the small angle approximation for the sine of the angle (i.e.
                    // the magnitude would be sine(angularLimit) * projection.magnitude
                    // but we approximate sine(angularLimit) to angularLimit).
                    real maxMagnitude = angularLimit * projection.Magnitude();

                    if (angularMove[i] < -maxMagnitude)
                    {
                        real totalMove = angularMove[i] + linearMove[i];
                        angularMove[i] = -maxMagnitude;
                        linearMove[i] = totalMove - angularMove[i];
                    }
                    else if (angularMove[i] > maxMagnitude)
                    {
                        real totalMove = angularMove[i] + linearMove[i];
                        angularMove[i] = maxMagnitude;
                        linearMove[i] = totalMove - angularMove[i];
                    }

                    // We have the linear amount of movement required by turning
                    // the rigid body (in angularMove[i]). We now need to
                    // calculate the desired rotation to achieve that.
                    if (angularMove[i] == 0)
                    {
                        // Easy case - no angular movement means no rotation.
                        angularChange[i].Clear();
                    }
                    else
                    {
                        // Work out the direction we'd like to rotate in.
                        MyVector3 targetAngularDirection =
                            relativeContactPosition[i].VectorProduct(contactNormal);

                        Matrix3 inverseInertiaTensor = new Matrix3();
                        body[i].getInverseInertiaTensorWorld(inverseInertiaTensor);

                        // Work out the direction we'd need to rotate to achieve that
                        angularChange[i] =
                            inverseInertiaTensor.transform(targetAngularDirection) *
                            (angularMove[i] / angularInertia[i]);
                    }

                    // Velocity change is easier - it is just the linear movement
                    // along the contact normal.
                    linearChange[i] = contactNormal * linearMove[i];

                    // Now we can start to apply the values we've calculated.
                    // Apply the linear movement
                    MyVector3 pos = new MyVector3();
                    body[i].getPosition(pos);
                    pos.AddScaledVector(contactNormal, linearMove[i]);
                    body[i].setPosition(pos);

                    // And the change in orientation
                    MyQuaternion q = new MyQuaternion();
                    body[i].getOrientation(q);
                    q.addScaledVector(angularChange[i], ((real)1.0));
                    body[i].setOrientation(q);

                    // We need to calculate the derived data for any body that is
                    // asleep, so that the changes are reflected in the object's
                    // data. Otherwise the resolution will not change the position
                    // of the object, and the next collision detection round will
                    // have the same penetration.
                    if (!body[i].getAwake()) body[i].calculateDerivedData();
                }
        }

        //DEV_NOTE not implemented in cyclone
        /**
         * Calculates the impulse needed to resolve this contact,
         * given that the contact has no friction. A pair of inertia
         * tensors - one for each contact object - is specified to
         * save calculation time: the calling function has access to
         * these anyway.
         */
        public MyVector3 calculateFrictionlessImpulse(Matrix3[] inverseInertiaTensor)
        {
            MyVector3 impulseContact = new MyVector3();

            // Build a vector that shows the change in velocity in
            // world space for a unit impulse in the direction of the contact
            // normal.
            MyVector3 deltaVelWorld = relativeContactPosition[0] % contactNormal;
            deltaVelWorld = inverseInertiaTensor[0].transform(deltaVelWorld);
            deltaVelWorld = deltaVelWorld % relativeContactPosition[0];

            // Work out the change in velocity in contact coordiantes.
            real deltaVelocity = deltaVelWorld * contactNormal;

            // Add the linear component of velocity change
            deltaVelocity += body[0].getInverseMass();

            // Check if we need to the second body's data
            if (body[1] != null)
            {
                // Go through the same transformation sequence again
                deltaVelWorld = relativeContactPosition[1] % contactNormal;
                deltaVelWorld = inverseInertiaTensor[1].transform(deltaVelWorld);
                deltaVelWorld = deltaVelWorld % relativeContactPosition[1];

                // Add the change in velocity due to rotation
                deltaVelocity += deltaVelWorld * contactNormal;

                // Add the change in velocity due to linear motion
                deltaVelocity += body[1].getInverseMass();
            }

            // Calculate the required size of the impulse
            impulseContact.x = desiredDeltaVelocity / deltaVelocity;
            impulseContact.y = 0;
            impulseContact.z = 0;
            return impulseContact;
        }

        //DEV_NOTE not implemented in cyclone
        /**
         * Calculates the impulse needed to resolve this contact,
         * given that the contact has a non-zero coefficient of
         * friction. A pair of inertia tensors - one for each contact
         * object - is specified to save calculation time: the calling
         * function has access to these anyway.
         */
        public MyVector3 calculateFrictionImpulse(Matrix3[] inverseInertiaTensor)
        {
            MyVector3 impulseContact = new MyVector3();
            real inverseMass = body[0].getInverseMass();

            // The equivalent of a cross product in matrices is multiplication
            // by a skew symmetric matrix - we build the matrix for converting
            // between linear and angular quantities.
            Matrix3 impulseToTorque = new Matrix3();
            impulseToTorque.setSkewSymmetric(relativeContactPosition[0]);

            // Build the matrix to convert contact impulse to change in velocity
            // in world coordinates.
            Matrix3 deltaVelWorld = impulseToTorque;
            deltaVelWorld *= inverseInertiaTensor[0];
            deltaVelWorld *= impulseToTorque;
            deltaVelWorld *= -1;

            // Check if we need to add body two's data
            if (body[1] != null)
            {
                // Set the cross product matrix
                impulseToTorque.setSkewSymmetric(relativeContactPosition[1]);

                // Calculate the velocity change matrix
                Matrix3 deltaVelWorld2 = impulseToTorque;
                deltaVelWorld2 *= inverseInertiaTensor[1];
                deltaVelWorld2 *= impulseToTorque;
                deltaVelWorld2 *= -1;

                // Add to the total delta velocity.
                deltaVelWorld += deltaVelWorld2;

                // Add to the inverse mass
                inverseMass += body[1].getInverseMass();
            }

            // Do a change of basis to convert into contact coordinates.
            Matrix3 deltaVelocity = contactToWorld.transpose();
            deltaVelocity *= deltaVelWorld;
            deltaVelocity *= contactToWorld;

            // Add in the linear velocity change
            deltaVelocity.data[0] += inverseMass;
            deltaVelocity.data[4] += inverseMass;
            deltaVelocity.data[8] += inverseMass;

            // Invert to get the impulse needed per unit velocity
            Matrix3 impulseMatrix = deltaVelocity.inverse();

            // Find the target velocities to kill
            MyVector3 velKill = new MyVector3(desiredDeltaVelocity,
        -contactVelocity.y,
        -contactVelocity.z);

            // Find the impulse to kill target velocities
            impulseContact = impulseMatrix.transform(velKill);

            // Check for exceeding friction
            real planarImpulse = Math.Sqrt(
                impulseContact.y * impulseContact.y +
                impulseContact.z * impulseContact.z
                );
            if (planarImpulse > impulseContact.x * friction)
            {
                // We need to use dynamic friction
                impulseContact.y /= planarImpulse;
                impulseContact.z /= planarImpulse;

                impulseContact.x = deltaVelocity.data[0] +
                    deltaVelocity.data[1] * friction * impulseContact.y +
                    deltaVelocity.data[2] * friction * impulseContact.z;
                impulseContact.x = desiredDeltaVelocity / impulseContact.x;
                impulseContact.y *= friction * impulseContact.x;
                impulseContact.z *= friction * impulseContact.x;
            }
            return impulseContact;
        }



       

    };
    /**
   * The contact resolution routine. One resolver instance
   * can be shared for the whole simulation, as long as you need
   * roughly the same parameters each time (which is normal).
   *
   * @section algorithm Resolution Algorithm
   *
   * The resolver uses an iterative satisfaction algorithm; it loops
   * through each contact and tries to resolve it. Each contact is
   * resolved locally, which may in turn put other contacts in a worse
   * position. The algorithm then revisits other contacts and repeats
   * the process up to a specified iteration limit. It can be proved
   * that given enough iterations, the simulation will get to the
   * correct result. As with all approaches, numerical stability can
   * cause problems that make a correct resolution impossible.
   *
   * @subsection strengths Strengths
   *
   * This algorithm is very fast, much faster than other physics
   * approaches. Even using many more iterations than there are
   * contacts, it will be faster than global approaches.
   *
   * Many global algorithms are unstable under high friction, this
   * approach is very robust indeed for high friction and low
   * restitution values.
   *
   * The algorithm produces visually believable behaviour. Tradeoffs
   * have been made to err on the side of visual realism rather than
   * computational expense or numerical accuracy.
   *
   * @subsection weaknesses Weaknesses
   *
   * The algorithm does not cope well with situations with many
   * inter-related contacts: stacked boxes, for example. In this
   * case the simulation may appear to jiggle slightly, which often
   * dislodges a box from the stack, allowing it to collapse.
   *
   * Another issue with the resolution mechanism is that resolving
   * one contact may make another contact move sideways against
   * friction, because each contact is handled independently, this
   * friction is not taken into account. If one object is pushing
   * against another, the pushed object may move across its support
   * without friction, even though friction is set between those bodies.
   *
   * In general this resolver is not suitable for stacks of bodies,
   * but is perfect for handling impact, explosive, and flat resting
   * situations.
   */
    class ContactResolver
    {

        /**
         * Holds the number of iterations to perform when resolving
         * velocity.
         */
        protected uint velocityIterations;

        /**
         * Holds the number of iterations to perform when resolving
         * position.
         */
        protected uint positionIterations;

        /**
         * To avoid instability velocities smaller
         * than this value are considered to be zero. Too small and the
         * simulation may be unstable, too large and the bodies may
         * interpenetrate visually. A good starting point is the default
         * of 0.01.
         */
        protected real velocityEpsilon;

        /**
         * To avoid instability penetrations
         * smaller than this value are considered to be not interpenetrating.
         * Too small and the simulation may be unstable, too large and the
         * bodies may interpenetrate visually. A good starting point is
         * the default of0.01.
         */
        protected real positionEpsilon;


        /**
         * Stores the number of velocity iterations used in the
         * last call to resolve contacts.
         */
        public uint velocityIterationsUsed;

        /**
         * Stores the number of position iterations used in the
         * last call to resolve contacts.
         */
        public uint positionIterationsUsed;


        /**
         * Keeps track of whether the internal settings are valid.
         */
        private bool validSettings;


        /**
         * Creates a new contact resolver with the given number of iterations
         * per resolution call, and optional epsilon values.
         */
        public ContactResolver(uint iterations,
        real velocityEpsilon = (real)0.01,
        real positionEpsilon = (real)0.01)
        {
            setIterations(iterations, iterations);
            setEpsilon(velocityEpsilon, positionEpsilon);
        }

        /**
         * Creates a new contact resolver with the given number of iterations
         * for each kind of resolution, and optional epsilon values.
         */
        public ContactResolver(uint velocityIterations,
            uint positionIterations,
            real velocityEpsilon = (real)0.01,
            real positionEpsilon = (real)0.01)
        {
            setIterations(velocityIterations);
            setEpsilon(velocityEpsilon, positionEpsilon);
        }

        /**
         * Returns true if the resolver has valid settings and is ready to go.
         */
        public bool isValid()
        {
            return (velocityIterations > 0) &&
                   (positionIterations > 0) &&
                   (positionEpsilon >= 0.0f) &&
                   (positionEpsilon >= 0.0f);
        }

        /**
         * Sets the number of iterations for each resolution stage.
         */
        public void setIterations(uint velocityIterations,
                           uint positionIterations)
        {
            this.velocityIterations = velocityIterations;
            this.positionIterations = positionIterations;
        }

        /**
         * Sets the number of iterations for both resolution stages.
         */
        public void setIterations(uint iterations)
        {
            setIterations(iterations, iterations);
        }

        /**
         * Sets the tolerance value for both velocity and position.
         */
        public void setEpsilon(real velocityEpsilon,
                        real positionEpsilon)
        {
            this.velocityEpsilon = velocityEpsilon;
            this.positionEpsilon = positionEpsilon;
        }

        /**
         * Resolves a set of contacts for both penetration and velocity.
         *
         * Contacts that cannot interact with
         * each other should be passed to separate calls to resolveContacts,
         * as the resolution algorithm takes much longer for lots of
         * contacts than it does for the same number of contacts in small
         * sets.
         *
         * @param contactArray Pointer to an array of contact objects.
         *
         * @param numContacts The number of contacts in the array to resolve.
         *
         * @param numIterations The number of iterations through the
         * resolution algorithm. This should be at least the number of
         * contacts (otherwise some constraints will not be resolved -
         * although sometimes this is not noticable). If the iterations are
         * not needed they will not be used, so adding more iterations may
         * not make any difference. In some cases you would need millions
         * of iterations. Think about the number of iterations as a bound:
         * if you specify a large number, sometimes the algorithm WILL use
         * it, and you may drop lots of frames.
         *
         * @param duration The duration of the previous integration step.
         * This is used to compensate for forces applied.
         */
        public void resolveContacts(Contact[] contacts,
            uint numContacts,
            real duration)
        {
            // Make sure we have something to do.
            if (numContacts == 0) return;
            if (!isValid()) return;

            // Prepare the contacts for processing
            prepareContacts(contacts, numContacts, duration);

            // Resolve the interpenetration problems with the contacts.
            adjustPositions(contacts, numContacts, duration);

            // Resolve the velocity problems with the contacts.
            adjustVelocities(contacts, numContacts, duration);
        }


        /**
         * Sets up contacts ready for processing. This makes sure their
         * internal data is configured correctly and the correct set of bodies
         * is made alive.
         */
        protected void prepareContacts(Contact[] contacts, uint numContacts,
        real duration)
        {
            // Generate contact velocity and axis information.
            Contact lastContact = contacts[numContacts];
            for (int i = 0; i < numContacts; i++)
            {
                Contact contact = contacts[i];
                // Calculate the internal contact data (inertia, basis, etc).
                contact.calculateInternals(duration);
            }
        }

        /**
         * Resolves the velocity issues with the given array of constraints,
         * using the given number of iterations.
         */
        protected void adjustVelocities(Contact[] c,
            uint numContacts,
            real duration)
        {
            MyVector3[] velocityChange = new MyVector3[2], rotationChange = new MyVector3[2];
            MyVector3 deltaVel;

            // iteratively handle impacts in order of severity.
            velocityIterationsUsed = 0;
            while (velocityIterationsUsed < velocityIterations)
            {
                // Find contact with maximum magnitude of probable velocity change.
                real max = velocityEpsilon;
                uint index = numContacts;
                for (uint i = 0; i < numContacts; i++)
                {
                    if (c[i].desiredDeltaVelocity > max)
                    {
                        max = c[i].desiredDeltaVelocity;
                        index = i;
                    }
                }
                if (index == numContacts) break;

                // Match the awake state at the contact
                c[index].matchAwakeState();

                // Do the resolution on the contact that came out top.
                c[index].applyVelocityChange(velocityChange, rotationChange);

                // With the change in velocity of the two bodies, the update of
                // contact velocities means that some of the relative closing
                // velocities need recomputing.
                for (uint i = 0; i < numContacts; i++)
                {
                    // Check each body in the contact
                    for (uint b = 0; b < 2; b++) if (c[i].body[b] != null)
                        {
                            // Check for a match with each body in the newly
                            // resolved contact
                            for (uint d = 0; d < 2; d++)
                            {
                                if (c[i].body[b] == c[index].body[d])
                                {
                                    deltaVel = velocityChange[d] +
                                        rotationChange[d].VectorProduct(
                                            c[i].relativeContactPosition[b]);

                                    // The sign of the change is negative if we're dealing
                                    // with the second body in a contact.
                                    c[i].contactVelocity +=
                                        c[i].contactToWorld.transformTranspose(deltaVel)
                                        * (b > 0 ? -1 : 1);
                                    c[i].calculateDesiredDeltaVelocity(duration);
                                }
                            }
                        }
                }
                velocityIterationsUsed++;
            }
        }

        /**
         * Resolves the positional issues with the given array of constraints,
         * using the given number of iterations.
         */
        protected void adjustPositions(Contact[] c,
            uint numContacts,
            real duration)
        {
            uint i, index;
            MyVector3[] linearChange = new MyVector3[2], angularChange = new MyVector3[2];
            real max;
            MyVector3 deltaPosition;

            // iteratively resolve interpenetrations in order of severity.
            positionIterationsUsed = 0;
            while (positionIterationsUsed < positionIterations)
            {
                // Find biggest penetration
                max = positionEpsilon;
                index = numContacts;
                for (i = 0; i < numContacts; i++)
                {
                    if (c[i].penetration > max)
                    {
                        max = c[i].penetration;
                        index = i;
                    }
                }
                if (index == numContacts) break;

                // Match the awake state at the contact
                c[index].matchAwakeState();

                // Resolve the penetration.
                c[index].applyPositionChange(
                    linearChange,
                    angularChange,
                    max);

                // Again this action may have changed the penetration of other
                // bodies, so we update contacts.
                for (i = 0; i < numContacts; i++)
                {
                    // Check each body in the contact
                    for (uint b = 0; b < 2; b++) if (c[i].body[b] != null)
                        {
                            // Check for a match with each body in the newly
                            // resolved contact
                            for (uint d = 0; d < 2; d++)
                            {
                                if (c[i].body[b] == c[index].body[d])
                                {
                                    deltaPosition = linearChange[d] +
                                        angularChange[d].VectorProduct(
                                            c[i].relativeContactPosition[b]);

                                    // The sign of the change is positive if we're
                                    // dealing with the second body in a contact
                                    // and negative otherwise (because we're
                                    // subtracting the resolution)..
                                    c[i].penetration +=
                                        deltaPosition.ScalarProduct(c[i].contactNormal)
                                        * (b > 0 ? 1 : -1);
                                }
                            }
                        }
                }
                positionIterationsUsed++;
            }
        }

    };
    /**
     * This is the basic polymorphic interface for contact generators
     * applying to rigid bodies.
     */
    class ContactGenerator
    {

        /**
         * Fills the given contact structure with the generated
         * contact. The contact pointer should point to the first
         * available contact in a contact array, where limit is the
         * maximum number of contacts in the array that can be written
         * to. The method returns the number of contacts that have
         * been written.
         */
        public virtual uint addContact(Contact contact, uint limit)
        {
            return 0;
        }
    };

}
