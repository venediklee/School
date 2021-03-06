using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace cyclone
{
    using real = System.Double;









    /**
     * A rigid body is the basic simulation object in the physics
     * core.
     *
     * It has position and orientation data, along with first
     * derivatives. It can be integrated forward through time, and
     * have forces, torques and impulses (linear or angular) applied
     * to it. The rigid body manages its state and allows access
     * through a set of methods.
     *
     * A ridid body contains 64 words (the size of which is given
     * by the precision: sizeof(real)). It contains no virtual
     * functions, so should take up exactly 64 words in memory. Of
     * this total 15 words are padding, distributed among the
     * MyVector3 data members.
     */
    class RigidBody
    {
        public static readonly real sleepEpsilon = (real)0.3;
        //public:

        // ... Other RigidBody code as before ...

        /*
 * --------------------------------------------------------------------------
 * INTERNAL OR HELPER FUNCTIONS:
 * --------------------------------------------------------------------------
 */

        /**
         * Internal function that checks the validity of an inverse inertia tensor.
         */
        static void _checkInverseInertiaTensor(Matrix3 iitWorld)
        {
            // TODO: Perform a validity check in an assert.
        }

        /**
         * Internal function to do an intertia tensor transform by a MyQuaternion.
         * Note that the implementation of this function was created by an
         * automated code-generator and optimizer.
         */
        static void _transformInertiaTensor(Matrix3 iitWorld,
                                                    MyQuaternion q,
                                                    Matrix3 iitBody,
                                                    Matrix4 rotmat)
        {
            real t4 = rotmat.data[0] * iitBody.data[0] +
                rotmat.data[1] * iitBody.data[3] +
                rotmat.data[2] * iitBody.data[6];
            real t9 = rotmat.data[0] * iitBody.data[1] +
                rotmat.data[1] * iitBody.data[4] +
                rotmat.data[2] * iitBody.data[7];
            real t14 = rotmat.data[0] * iitBody.data[2] +
                rotmat.data[1] * iitBody.data[5] +
                rotmat.data[2] * iitBody.data[8];
            real t28 = rotmat.data[4] * iitBody.data[0] +
                rotmat.data[5] * iitBody.data[3] +
                rotmat.data[6] * iitBody.data[6];
            real t33 = rotmat.data[4] * iitBody.data[1] +
                rotmat.data[5] * iitBody.data[4] +
                rotmat.data[6] * iitBody.data[7];
            real t38 = rotmat.data[4] * iitBody.data[2] +
                rotmat.data[5] * iitBody.data[5] +
                rotmat.data[6] * iitBody.data[8];
            real t52 = rotmat.data[8] * iitBody.data[0] +
                rotmat.data[9] * iitBody.data[3] +
                rotmat.data[10] * iitBody.data[6];
            real t57 = rotmat.data[8] * iitBody.data[1] +
                rotmat.data[9] * iitBody.data[4] +
                rotmat.data[10] * iitBody.data[7];
            real t62 = rotmat.data[8] * iitBody.data[2] +
                rotmat.data[9] * iitBody.data[5] +
                rotmat.data[10] * iitBody.data[8];

            iitWorld.data[0] = t4 * rotmat.data[0] +

                t9 * rotmat.data[1] +

                t14 * rotmat.data[2];
            iitWorld.data[1] = t4 * rotmat.data[4] +

                t9 * rotmat.data[5] +

                t14 * rotmat.data[6];
            iitWorld.data[2] = t4 * rotmat.data[8] +

                t9 * rotmat.data[9] +

                t14 * rotmat.data[10];
            iitWorld.data[3] = t28 * rotmat.data[0] +

                t33 * rotmat.data[1] +

                t38 * rotmat.data[2];
            iitWorld.data[4] = t28 * rotmat.data[4] +

                t33 * rotmat.data[5] +

                t38 * rotmat.data[6];
            iitWorld.data[5] = t28 * rotmat.data[8] +

                t33 * rotmat.data[9] +

                t38 * rotmat.data[10];
            iitWorld.data[6] = t52 * rotmat.data[0] +

                t57 * rotmat.data[1] +

                t62 * rotmat.data[2];
            iitWorld.data[7] = t52 * rotmat.data[4] +

                t57 * rotmat.data[5] +

                t62 * rotmat.data[6];
            iitWorld.data[8] = t52 * rotmat.data[8] +

                t57 * rotmat.data[9] +

                t62 * rotmat.data[10];
        }


        /**
     * Inline function that creates a transform matrix from a
     * position and orientation.
     */
        static void _calculateTransformMatrix(Matrix4 transformMatrix,
                                                      MyVector3 position,
                                                      MyQuaternion orientation)
        {
            transformMatrix.data[0] = 1 - 2 * orientation.j * orientation.j -
                2 * orientation.k * orientation.k;
            transformMatrix.data[1] = 2 * orientation.i * orientation.j -
                2 * orientation.r * orientation.k;
            transformMatrix.data[2] = 2 * orientation.i * orientation.k +
                2 * orientation.r * orientation.j;
            transformMatrix.data[3] = position.x;

            transformMatrix.data[4] = 2 * orientation.i * orientation.j +
                2 * orientation.r * orientation.k;
            transformMatrix.data[5] = 1 - 2 * orientation.i * orientation.i -
                2 * orientation.k * orientation.k;
            transformMatrix.data[6] = 2 * orientation.j * orientation.k -
                2 * orientation.r * orientation.i;
            transformMatrix.data[7] = position.y;

            transformMatrix.data[8] = 2 * orientation.i * orientation.k -
                2 * orientation.r * orientation.j;
            transformMatrix.data[9] = 2 * orientation.j * orientation.k +
                2 * orientation.r * orientation.i;
            transformMatrix.data[10] = 1 - 2 * orientation.i * orientation.i -
                2 * orientation.j * orientation.j;
            transformMatrix.data[11] = position.z;
        }
        /*
     * --------------------------------------------------------------------------
     * FUNCTIONS DECLARED IN HEADER:
     * --------------------------------------------------------------------------
     */



        /**
         * @name Characteristic Data and State
         *
         * This data holds the state of the rigid body. There are two
         * sets of data: characteristics and state.
         *
         * Characteristics are properties of the rigid body
         * independent of its current kinematic situation. This
         * includes mass, moment of inertia and damping
         * properties. Two identical rigid bodys will have the same
         * values for their characteristics.
         *
         * State includes all the characteristics and also includes
         * the kinematic situation of the rigid body in the current
         * simulation. By setting the whole state data, a rigid body's
         * exact game state can be replicated. Note that state does
         * not include any forces applied to the body. Two identical
         * rigid bodies in the same simulation will not share the same
         * state values.
         *
         * The state values make up the smallest set of independent
         * data for the rigid body. Other state data is calculated
         * from their current values. When state data is changed the
         * dependent values need to be updated: this can be achieved
         * either by integrating the simulation, or by calling the
         * calculateInternals function. This two stage process is used
         * because recalculating internals can be a costly process:
         * all state changes should be carried out at the same time,
         * allowing for a single call.
         *
         * @see calculateInternals
         */
        /*@{*/
        /**
         * Holds the inverse of the mass of the rigid body. It
         * is more useful to hold the inverse mass because
         * integration is simpler, and because in real time
         * simulation it is more useful to have bodies with
         * infinite mass (immovable) than zero mass
         * (completely unstable in numerical simulation).
         */
        protected real inverseMass;

        /**
         * Holds the inverse of the body's inertia tensor. The
         * inertia tensor provided must not be degenerate
         * (that would mean the body had zero inertia for
         * spinning along one axis). As long as the tensor is
         * finite, it will be invertible. The inverse tensor
         * is used for similar reasons to the use of inverse
         * mass.
         *
         * The inertia tensor, unlike the other variables that
         * define a rigid body, is given in body space.
         *
         * @see inverseMass
         */
        protected Matrix3 inverseInertiaTensor;

        /**
         * Holds the amount of damping applied to linear
         * motion.  Damping is required to remove energy added
         * through numerical instability in the integrator.
         */
        protected real linearDamping;

        /**
         * Holds the amount of damping applied to angular
         * motion.  Damping is required to remove energy added
         * through numerical instability in the integrator.
         */
        protected real angularDamping;

        /**
         * Holds the linear position of the rigid body in
         * world space.
         */
        protected MyVector3 position;

        /**
         * Holds the angular orientation of the rigid body in
         * world space.
         */
        protected MyQuaternion orientation;

        /**
         * Holds the linear velocity of the rigid body in
         * world space.
         */
        protected MyVector3 velocity;

        /**
         * Holds the angular velocity, or rotation, or the
         * rigid body in world space.
         */
        protected MyVector3 rotation;

        /*@}*/


        /**
         * @name Derived Data
         *
         * These data members hold information that is derived from
         * the other data in the class.
         */
        /*@{*/

        /**
         * Holds the inverse inertia tensor of the body in world
         * space. The inverse inertia tensor member is specified in
         * the body's local space.
         *
         * @see inverseInertiaTensor
         */
        protected Matrix3 inverseInertiaTensorWorld;

        /**
         * Holds the amount of motion of the body. This is a recency
         * weighted mean that can be used to put a body to sleap.
         */
        protected real motion;

        /**
         * A body can be put to sleep to avoid it being updated
         * by the integration functions or affected by collisions
         * with the world.
         */
        protected bool isAwake;

        /**
         * Some bodies may never be allowed to fall asleep.
         * User controlled bodies, for example, should be
         * always awake.
         */
        protected bool canSleep;

        /**
         * Holds a transform matrix for converting body space into
         * world space and vice versa. This can be achieved by calling
         * the getPointIn*Space functions.
         *
         * @see getPointInLocalSpace
         * @see getPointInWorldSpace
         * @see getTransform
         */
        protected Matrix4 transformMatrix;

        /*@}*/


        /**
         * @name Force and Torque Accumulators
         *
         * These data members store the current force, torque and
         * acceleration of the rigid body. Forces can be added to the
         * rigid body in any order, and the class decomposes them into
         * their constituents, accumulating them for the next
         * simulation step. At the simulation step, the accelerations
         * are calculated and stored to be applied to the rigid body.
         */
        /*@{*/

        /**
         * Holds the accumulated force to be applied at the next
         * integration step.
         */
        protected MyVector3 forceAccum;

        /**
         * Holds the accumulated torque to be applied at the next
         * integration step.
         */
        protected MyVector3 torqueAccum;

        /**
          * Holds the acceleration of the rigid body.  This value
          * can be used to set acceleration due to gravity (its primary
          * use), or any other constant acceleration.
          */
        protected MyVector3 acceleration;

        /**
         * Holds the linear acceleration of the rigid body, for the
         * previous frame.
         */
        protected MyVector3 lastFrameAcceleration;

        /*@}*/


        /**
         * @name Constructor and Destructor
         *
         * There are no data members in the rigid body class that are
         * created on the heap. So all data storage is handled
         * automatically.
         */
        /*@{*/

        /*@}*/


        /**
         * @name Integration and Simulation Functions
         *
         * These functions are used to simulate the rigid body's
         * motion over time. A normal application sets up one or more
         * rigid bodies, applies permanent forces (i.e. gravity), then
         * adds transient forces each frame, and integrates, prior to
         * rendering.
         *
         * Currently the only integration function provided is the
         * first order Newton Euler method.
         */
        /*@{*/

        /**
         * Calculates internal data from state data. This should be called
         * after the body's state is altered directly (it is called
         * automatically during integration). If you change the body's state
         * and then intend to integrate before querying any data (such as
         * the transform matrix), then you can ommit this step.
         */
        public void calculateDerivedData()
        {
            orientation.normalise();

            // Calculate the transform matrix for the body.
            _calculateTransformMatrix(transformMatrix, position, orientation);

            // Calculate the inertiaTensor in world space.
            _transformInertiaTensor(inverseInertiaTensorWorld,
                orientation,
                inverseInertiaTensor,
                transformMatrix);
        }

        /**
         * Integrates the rigid body forward in time by the given amount.
         * This function uses a Newton-Euler integration method, which is a
         * linear approximation to the correct integral. For this reason it
         * may be inaccurate in some cases.
         */
        public void integrate(real duration)
        {
            if (!isAwake) return;

            // Calculate linear acceleration from force inputs.
            lastFrameAcceleration = acceleration;
            lastFrameAcceleration.AddScaledVector(forceAccum, inverseMass);

            // Calculate angular acceleration from torque inputs.
            MyVector3 angularAcceleration =
                inverseInertiaTensorWorld.transform(torqueAccum);

            // Adjust velocities
            // Update linear velocity from both acceleration and impulse.
            velocity.AddScaledVector(lastFrameAcceleration, duration);

            // Update angular velocity from both acceleration and impulse.
            rotation.AddScaledVector(angularAcceleration, duration);

            // Impose drag.
            velocity *= Mathf.Pow((float)linearDamping, (float)duration);
            rotation *= Mathf.Pow((float)angularDamping, (float)duration);

            // Adjust positions
            // Update linear position.
            position.AddScaledVector(velocity, duration);

            // Update angular position.
            orientation.addScaledVector(rotation, duration);

            // Normalise the orientation, and update the matrices with the new
            // position and orientation
            calculateDerivedData();

            // Clear accumulators.
            clearAccumulators();

            // Update the kinetic energy store, and possibly put the body to
            // sleep.
            if (canSleep)
            {
                real currentMotion = velocity.ScalarProduct(velocity) +
                    rotation.ScalarProduct(rotation);

                real bias = Mathf.Pow(0.5f, (float)duration);
                motion = bias * motion + (1 - bias) * currentMotion;

                if (motion < sleepEpsilon) setAwake(false);
                else if (motion > 10 * sleepEpsilon) motion = 10 * sleepEpsilon;
            }
        }

        /*@}*/


        /**
         * @name Accessor Functions for the Rigid Body's State
         *
         * These functions provide access to the rigid body's
         * characteristics or state. These data can be accessed
         * individually, or en masse as an array of values
         * (e.g. getCharacteristics, getState). When setting new data,
         * make sure the calculateInternals function, or an
         * integration routine, is called before trying to get data
         * from the body, since the class contains a number of
         * dependent values that will need recalculating.
         */
        /*@{*/

        /**
         * Sets the mass of the rigid body.
         *
         * @param mass The new mass of the body. This may not be zero.
         * Small masses can produce unstable rigid bodies under
         * simulation.
         *
         * @warning This invalidates internal data for the rigid body.
         * Either an integration function, or the calculateInternals
         * function should be called before trying to get any settings
         * from the rigid body.
         */
        public void setMass(real mass)
        {
            Debug.Assert(mass != 0);
            this.inverseMass = ((real)1.0) / mass;
        }

        /**
         * Gets the mass of the rigid body.
         *
         * @return The current mass of the rigid body.
         */
        public real getMass()
        {
            if (this.inverseMass == 0)
            {
                return real.MaxValue;
            }
            else
            {
                return ((real)1.0) / inverseMass;
            }
        }

        /**
         * Sets the inverse mass of the rigid body.
         *
         * @param inverseMass The new inverse mass of the body. This
         * may be zero, for a body with infinite mass
         * (i.e. unmovable).
         *
         * @warning This invalidates internal data for the rigid body.
         * Either an integration function, or the calculateInternals
         * function should be called before trying to get any settings
         * from the rigid body.
         */
        public void setInverseMass(real inverseMass)
        {
            this.inverseMass = inverseMass;
        }

        /**
         * Gets the inverse mass of the rigid body.
         *
         * @return The current inverse mass of the rigid body.
         */
        public real getInverseMass()
        {
            return inverseMass;
        }

        /**
         * Returns true if the mass of the body is not-infinite.
         */
        public bool hasFiniteMass()
        {
            return inverseMass >= 0.0f;
        }

        /**
         * Sets the intertia tensor for the rigid body.
         *
         * @param inertiaTensor The inertia tensor for the rigid
         * body. This must be a full rank matrix and must be
         * invertible.
         *
         * @warning This invalidates internal data for the rigid body.
         * Either an integration function, or the calculateInternals
         * function should be called before trying to get any settings
         * from the rigid body.
         */
        public void setInertiaTensor(Matrix3 inertiaTensor)
        {
            inverseInertiaTensor.setInverse(inertiaTensor);
            _checkInverseInertiaTensor(inverseInertiaTensor);
        }

        /**
         * Copies the current inertia tensor of the rigid body into
         * the given matrix.
         *
         * @param inertiaTensor A pointer to a matrix to hold the
         * current inertia tensor of the rigid body. The inertia
         * tensor is expressed in the rigid body's local space.
         */
        public void getInertiaTensor(Matrix3 inertiaTensor)
        {
            inertiaTensor.setInverse(inverseInertiaTensor);
        }

        /**
         * Gets a copy of the current inertia tensor of the rigid body.
         *
         * @return A new matrix containing the current intertia
         * tensor. The inertia tensor is expressed in the rigid body's
         * local space.
         */
        public Matrix3 getInertiaTensor()
        {
            Matrix3 it = new Matrix3();
            getInertiaTensor(it);
            return it;
        }


        /**
         * Copies the current inertia tensor of the rigid body into
         * the given matrix.
         *
         * @param inertiaTensor A pointer to a matrix to hold the
         * current inertia tensor of the rigid body. The inertia
         * tensor is expressed in world space.
         */
        public void getInertiaTensorWorld(Matrix3 inertiaTensor)
        {
            inertiaTensor.setInverse(inverseInertiaTensorWorld);
        }

        /**
         * Gets a copy of the current inertia tensor of the rigid body.
         *
         * @return A new matrix containing the current intertia
         * tensor. The inertia tensor is expressed in world space.
         */
        public Matrix3 getInertiaTensorWorld()
        {
            Matrix3 it = new Matrix3();
            getInertiaTensorWorld(it);
            return it;
        }

        /**
         * Sets the inverse intertia tensor for the rigid body.
         *
         * @param inverseInertiaTensor The inverse inertia tensor for
         * the rigid body. This must be a full rank matrix and must be
         * invertible.
         *
         * @warning This invalidates internal data for the rigid body.
         * Either an integration function, or the calculateInternals
         * function should be called before trying to get any settings
         * from the rigid body.
         */
        public void setInverseInertiaTensor(Matrix3 inverseInertiaTensor)
        {
            _checkInverseInertiaTensor(inverseInertiaTensor);
            this.inverseInertiaTensor = inverseInertiaTensor;
        }

        /**
         * Copies the current inverse inertia tensor of the rigid body
         * into the given matrix.
         *
         * @param inverseInertiaTensor A pointer to a matrix to hold
         * the current inverse inertia tensor of the rigid body. The
         * inertia tensor is expressed in the rigid body's local
         * space.
         */
        public void getInverseInertiaTensor(Matrix3 inverseInertiaTensor)
        {
            inverseInertiaTensor = this.inverseInertiaTensor;
        }

        /**
         * Gets a copy of the current inverse inertia tensor of the
         * rigid body.
         *
         * @return A new matrix containing the current inverse
         * intertia tensor. The inertia tensor is expressed in the
         * rigid body's local space.
         */
        public Matrix3 getInverseInertiaTensor()
        {
            return inverseInertiaTensor;
        }

        /**
         * Copies the current inverse inertia tensor of the rigid body
         * into the given matrix.
         *
         * @param inverseInertiaTensor A pointer to a matrix to hold
         * the current inverse inertia tensor of the rigid body. The
         * inertia tensor is expressed in world space.
         */
        public void getInverseInertiaTensorWorld(Matrix3 inverseInertiaTensor)
        {
            inverseInertiaTensor = inverseInertiaTensorWorld;
        }

        /**
         * Gets a copy of the current inverse inertia tensor of the
         * rigid body.
         *
         * @return A new matrix containing the current inverse
         * intertia tensor. The inertia tensor is expressed in world
         * space.
         */
        public Matrix3 getInverseInertiaTensorWorld()
        {
            return inverseInertiaTensorWorld;
        }

        /**
         * Sets both linear and angular damping in one function call.
         *
         * @param linearDamping The speed that velocity is shed from
         * the rigid body.
         *
         * @param angularDamping The speed that rotation is shed from
         * the rigid body.
         *
         * @see setLinearDamping
         * @see setAngularDamping
         */
        public void setDamping(real linearDamping, real angularDamping)
        {
            this.linearDamping = linearDamping;
            this.angularDamping = angularDamping;
        }

        /**
         * Sets the linear damping for the rigid body.
         *
         * @param linearDamping The speed that velocity is shed from
         * the rigid body.
         *
         * @see setAngularDamping
         */
        public void setLinearDamping(real linearDamping)
        {
            this.linearDamping = linearDamping;
        }

        /**
         * Gets the current linear damping value.
         *
         * @return The current linear damping value.
         */
        public real getLinearDamping()
        {
            return linearDamping;
        }

        /**
         * Sets the angular damping for the rigid body.
         *
         * @param angularDamping The speed that rotation is shed from
         * the rigid body.
         *
         * @see setLinearDamping
         */
        public void setAngularDamping(real angularDamping)
        {
            this.angularDamping = angularDamping;
        }

        /**
         * Gets the current angular damping value.
         *
         * @return The current angular damping value.
         */
        public real getAngularDamping()
        {
            return angularDamping;
        }

        /**
         * Sets the position of the rigid body.
         *
         * @param position The new position of the rigid body.
         */
        public void setPosition(MyVector3 position)
        {
            this.position = position;
        }

        /**
         * Sets the position of the rigid body by component.
         *
         * @param x The x coordinate of the new position of the rigid
         * body.
         *
         * @param y The y coordinate of the new position of the rigid
         * body.
         *
         * @param z The z coordinate of the new position of the rigid
         * body.
         */
        public void setPosition(real x, real y, real z)
        {
            position.x = x;
            position.y = y;
            position.z = z;
        }

        /**
         * Fills the given vector with the position of the rigid body.
         *
         * @param position A pointer to a vector into which to write
         * the position.
         */
        public void getPosition(MyVector3 position)
        {
            position = this.position;
        }

        /**
         * Gets the position of the rigid body.
         *
         * @return The position of the rigid body.
         */
        public MyVector3 getPosition()
        {
            return position;
        }

        /**
         * Sets the orientation of the rigid body.
         *
         * @param orientation The new orientation of the rigid body.
         *
         * @note The given orientation does not need to be normalised,
         * and can be zero. This function automatically constructs a
         * valid rotation MyQuaternion with (0,0,0,0) mapping to
         * (1,0,0,0).
         */
        public void setOrientation(MyQuaternion orientation)
        {
            this.orientation = orientation;
            this.orientation.normalise();
        }

        /**
         * Sets the orientation of the rigid body by component.
         *
         * @param r The real component of the rigid body's orientation
         * MyQuaternion.
         *
         * @param i The first complex component of the rigid body's
         * orientation MyQuaternion.
         *
         * @param j The second complex component of the rigid body's
         * orientation MyQuaternion.
         *
         * @param k The third complex component of the rigid body's
         * orientation MyQuaternion.
         *
         * @note The given orientation does not need to be normalised,
         * and can be zero. This function automatically constructs a
         * valid rotation MyQuaternion with (0,0,0,0) mapping to
         * (1,0,0,0).
         */
        public void setOrientation(real r, real i, real j, real k)
        {
            orientation.r = r;
            orientation.i = i;
            orientation.j = j;
            orientation.k = k;
            orientation.normalise();
        }

        /**
         * Fills the given MyQuaternion with the current value of the
         * rigid body's orientation.
         *
         * @param orientation A pointer to a MyQuaternion to receive the
         * orientation data.
         */
        public void getOrientation(MyQuaternion orientation)
        {
            orientation = this.orientation;
        }

        /**
         * Gets the orientation of the rigid body.
         *
         * @return The orientation of the rigid body.
         */
        public MyQuaternion getOrientation()
        {
            return orientation;
        }

        /**
         * Fills the given matrix with a transformation representing
         * the rigid body's orientation.
         *
         * @note Transforming a direction vector by this matrix turns
         * it from the body's local space to world space.
         *
         * @param matrix A pointer to the matrix to fill.
         */
        public void getOrientation(Matrix3 matrix)
        {
            getOrientation(matrix.data);
        }

        /**
         * Fills the given matrix data structure with a transformation
         * representing the rigid body's orientation.
         *
         * @note Transforming a direction vector by this matrix turns
         * it from the body's local space to world space.
         *
         * @param matrix A pointer to the matrix to fill.
         */
        public void getOrientation(real[] matrix)
        {
            matrix[0] = transformMatrix.data[0];
            matrix[1] = transformMatrix.data[1];
            matrix[2] = transformMatrix.data[2];

            matrix[3] = transformMatrix.data[4];
            matrix[4] = transformMatrix.data[5];
            matrix[5] = transformMatrix.data[6];

            matrix[6] = transformMatrix.data[8];
            matrix[7] = transformMatrix.data[9];
            matrix[8] = transformMatrix.data[10];
        }

        /**
         * Fills the given matrix with a transformation representing
         * the rigid body's position and orientation.
         *
         * @note Transforming a vector by this matrix turns it from
         * the body's local space to world space.
         *
         * @param transform A pointer to the matrix to fill.
         */
        public void getTransform(Matrix4 transform)
        {
            transform = new Matrix4(transformMatrix.data);
        }

        /**
         * Fills the given matrix data structure with a
         * transformation representing the rigid body's position and
         * orientation.
         *
         * @note Transforming a vector by this matrix turns it from
         * the body's local space to world space.
         *
         * @param matrix A pointer to the matrix to fill.
         */
        public void getTransform(real[] matrix)
        {
            matrix = transformMatrix.data;
            matrix[12] = matrix[13] = matrix[14] = 0;
            matrix[15] = 1;
        }

        /**
         * Gets a transformation representing the rigid body's
         * position and orientation.
         *
         * @note Transforming a vector by this matrix turns it from
         * the body's local space to world space.
         *
         * @return The transform matrix for the rigid body.
         */
        public Matrix4 getTransform()
        {
            return transformMatrix;
        }

        /**
         * Converts the given point from world space into the body's
         * local space.
         *
         * @param point The point to covert, given in world space.
         *
         * @return The converted point, in local space.
         */
        public MyVector3 getPointInLocalSpace(MyVector3 point)
        {
            return transformMatrix.transformInverse(point);
        }

        /**
         * Converts the given point from world space into the body's
         * local space.
         *
         * @param point The point to covert, given in local space.
         *
         * @return The converted point, in world space.
         */
        public MyVector3 getPointInWorldSpace(MyVector3 point)
        {
            return transformMatrix.transform(point);
        }

        /**
         * Converts the given direction from world space into the
         * body's local space.
         *
         * @note When a direction is converted between frames of
         * reference, there is no translation required.
         *
         * @param direction The direction to covert, given in world
         * space.
         *
         * @return The converted direction, in local space.
         */
        public MyVector3 getDirectionInLocalSpace(MyVector3 direction)
        {
            return transformMatrix.transformInverseDirection(direction);
        }

        /**
         * Converts the given direction from world space into the
         * body's local space.
         *
         * @note When a direction is converted between frames of
         * reference, there is no translation required.
         *
         * @param direction The direction to covert, given in local
         * space.
         *
         * @return The converted direction, in world space.
         */
        public MyVector3 getDirectionInWorldSpace(MyVector3 direction)
        {
            return transformMatrix.transformDirection(direction);
        }

        /**
         * Sets the velocity of the rigid body.
         *
         * @param velocity The new velocity of the rigid body. The
         * velocity is given in world space.
         */
        public void setVelocity(MyVector3 velocity)
        {
            this.velocity = velocity;
        }

        /**
         * Sets the velocity of the rigid body by component. The
         * velocity is given in world space.
         *
         * @param x The x coordinate of the new velocity of the rigid
         * body.
         *
         * @param y The y coordinate of the new velocity of the rigid
         * body.
         *
         * @param z The z coordinate of the new velocity of the rigid
         * body.
         */
        public void setVelocity(real x, real y, real z)
        {
            velocity.x = x;
            velocity.y = y;
            velocity.z = z;
        }

        /**
         * Fills the given vector with the velocity of the rigid body.
         *
         * @param velocity A pointer to a vector into which to write
         * the velocity. The velocity is given in world local space.
         */
        public void getVelocity(MyVector3 velocity)
        {
            velocity = this.velocity;
        }

        /**
         * Gets the velocity of the rigid body.
         *
         * @return The velocity of the rigid body. The velocity is
         * given in world local space.
         */
        public MyVector3 getVelocity()
        {
            return velocity;
        }

        /**
         * Applies the given change in velocity.
         */
        public void addVelocity(MyVector3 deltaVelocity)
        {
            this.velocity += deltaVelocity;
        }

        /**
         * Sets the rotation of the rigid body.
         *
         * @param rotation The new rotation of the rigid body. The
         * rotation is given in world space.
         */
        public void setRotation(MyVector3 rotation)
        {
            this.rotation = rotation;
        }

        /**
         * Sets the rotation of the rigid body by component. The
         * rotation is given in world space.
         *
         * @param x The x coordinate of the new rotation of the rigid
         * body.
         *
         * @param y The y coordinate of the new rotation of the rigid
         * body.
         *
         * @param z The z coordinate of the new rotation of the rigid
         * body.
         */
        public void setRotation(real x, real y, real z)
        {
            rotation.x = x;
            rotation.y = y;
            rotation.z = z;
        }

        /**
         * Fills the given vector with the rotation of the rigid body.
         *
         * @param rotation A pointer to a vector into which to write
         * the rotation. The rotation is given in world local space.
         */
        public void getRotation(MyVector3 rotation)
        {
            rotation = this.rotation;
        }

        /**
         * Gets the rotation of the rigid body.
         *
         * @return The rotation of the rigid body. The rotation is
         * given in world local space.
         */
        public MyVector3 getRotation()
        {
            return rotation;
        }

        /**
         * Applies the given change in rotation.
         */
        public void addRotation(MyVector3 deltaRotation)
        {
            rotation += deltaRotation;
        }

        /**
         * Returns true if the body is awake and responding to
         * integration.
         *
         * @return The awake state of the body.
         */
        public bool getAwake()
        {
            return isAwake;
        }

        /**
         * Sets the awake state of the body. If the body is set to be
         * not awake, then its velocities are also cancelled, since
         * a moving body that is not awake can cause problems in the
         * simulation.
         *
         * @param awake The new awake state of the body.
         */
        public void setAwake(bool awake = true)
        {
            if (awake)
            {
                isAwake = true;

                // Add a bit of motion to avoid it falling asleep immediately.
                motion = sleepEpsilon * 2.0f;
            }
            else
            {
                isAwake = false;
                velocity.Clear();
                rotation.Clear();
            }
        }

        /**
         * Returns true if the body is allowed to go to sleep at
         * any time.
         */
        public bool getCanSleep()
        {
            return canSleep;
        }

        /**
         * Sets whether the body is ever allowed to go to sleep. Bodies
         * under the player's control, or for which the set of
         * transient forces applied each frame are not predictable,
         * should be kept awake.
         *
         * @param canSleep Whether the body can now be put to sleep.
         */
        public void setCanSleep(bool canSleep = true)
        {

            this.canSleep = canSleep;

            if (!canSleep && !isAwake) setAwake();
        }

        /*@}*/


        /**
         * @name Retrieval Functions for Dynamic Quantities
         *
         * These functions provide access to the acceleration
         * properties of the body. The acceleration is generated by
         * the simulation from the forces and torques applied to the
         * rigid body. Acceleration cannot be directly influenced, it
         * is set during integration, and represent the acceleration
         * experienced by the body of the previous simulation step.
         */
        /*@{*/

        /**
         * Fills the given vector with the current accumulated value
         * for linear acceleration. The acceleration accumulators
         * are set during the integration step. They can be read to
         * determine the rigid body's acceleration over the last
         * integration step. The linear acceleration is given in world
         * space.
         *
         * @param linearAcceleration A pointer to a vector to receive
         * the linear acceleration data.
         */
        public void getLastFrameAcceleration(MyVector3 linearAcceleration)
        {
            linearAcceleration = this.lastFrameAcceleration;
        }

        /**
         * Gets the current accumulated value for linear
         * acceleration. The acceleration accumulators are set during
         * the integration step. They can be read to determine the
         * rigid body's acceleration over the last integration
         * step. The linear acceleration is given in world space.
         *
         * @return The rigid body's linear acceleration.
         */
        public MyVector3 getLastFrameAcceleration()
        {
            return getLastFrameAcceleration();
        }

        /*@}*/


        /**
         * @name Force, Torque and Acceleration Set-up Functions
         *
         * These functions set up forces and torques to apply to the
         * rigid body.
         */
        /*@{*/

        /**
         * Clears the forces and torques in the accumulators. This will
         * be called automatically after each intergration step.
         */
        public void clearAccumulators()
        {

            forceAccum.Clear();
            torqueAccum.Clear();
        }

        /**
         * Adds the given force to centre of mass of the rigid body.
         * The force is expressed in world-coordinates.
         *
         * @param force The force to apply.
         */
        public void addForce(MyVector3 force)
        {
            forceAccum += force;
            isAwake = true;
        }

        /**
         * Adds the given force to the given point on the rigid body.
         * Both the force and the
         * application point are given in world space. Because the
         * force is not applied at the centre of mass, it may be split
         * into both a force and torque.
         *
         * @param force The force to apply.
         *
         * @param point The location at which to apply the force, in
         * world-coordinates.
         */
        public void addForceAtPoint(MyVector3 force, MyVector3 point)
        {

            // Convert to coordinates relative to center of mass.
            MyVector3 pt = point;
            pt -= position;

            forceAccum += force;
            torqueAccum += pt % force;

            isAwake = true;
        }
        /**
         * Adds the given force to the given point on the rigid body.
         * The direction of the force is given in world coordinates,
         * but the application point is given in body space. This is
         * useful for spring forces, or other forces fixed to the
         * body.
         *
         * @param force The force to apply.
         *
         * @param point The location at which to apply the force, in
         * body-coordinates.
         */
        public void addForceAtBodyPoint(MyVector3 force, MyVector3 point)
        {


            // Convert to coordinates relative to center of mass.
            MyVector3 pt = getPointInWorldSpace(point);
            addForceAtPoint(force, pt);
        }

        /**
         * Adds the given torque to the rigid body.
         * The force is expressed in world-coordinates.
         *
         * @param torque The torque to apply.
         */
        public void addTorque(MyVector3 torque)
        {
            torqueAccum += torque;
            isAwake = true;
        }

        /**
         * Sets the constant acceleration of the rigid body.
         *
         * @param acceleration The new acceleration of the rigid body.
         */
        public void setAcceleration(MyVector3 acceleration)
        {
            this.acceleration = acceleration;
        }

        /**
         * Sets the constant acceleration of the rigid body by component.
         *
         * @param x The x coordinate of the new acceleration of the rigid
         * body.
         *
         * @param y The y coordinate of the new acceleration of the rigid
         * body.
         *
         * @param z The z coordinate of the new acceleration of the rigid
         * body.
         */
        public void setAcceleration(real x, real y, real z)
        {
            acceleration.x = x;
            acceleration.y = y;
            acceleration.z = z;
        }

        /**
         * Fills the given vector with the acceleration of the rigid body.
         *
         * @param acceleration A pointer to a vector into which to write
         * the acceleration. The acceleration is given in world local space.
         */
        public void getAcceleration(MyVector3 acceleration)
        {
            acceleration = this.acceleration;
        }

        /**
         * Gets the acceleration of the rigid body.
         *
         * @return The acceleration of the rigid body. The acceleration is
         * given in world local space.
         */
        public MyVector3 getAcceleration()
        {
            return acceleration;
        }

        /*@}*/

    };
}
