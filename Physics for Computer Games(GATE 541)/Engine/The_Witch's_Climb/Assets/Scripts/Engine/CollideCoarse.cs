using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

//TODO fix this script
//namespace cyclone
//{
//    using real = System.Double;

//    /**
//     * Represents a bounding sphere that can be tested for overlap.
//     */
//    struct BoundingSphere
//    {
//        MyVector3 centre;
//        real radius;


//        /**
//         * Creates a new bounding sphere at the given centre and radius.
//         */
//        public BoundingSphere(MyVector3 centre, real radius)
//        {
//            this.centre = centre;
//            this.radius = radius;
//        }

//        /**
//         * Creates a bounding sphere to enclose the two given bounding
//         * spheres.
//         */
//        public BoundingSphere(BoundingSphere one, BoundingSphere two)
//        {
//            MyVector3 centreOffset = two.centre - one.centre;
//            real distance = centreOffset.SquareMagnitude();
//            real radiusDiff = two.radius - one.radius;

//            // Check if the larger sphere encloses the small one
//            if (radiusDiff * radiusDiff >= distance)
//            {
//                if (one.radius > two.radius)
//                {
//                    centre = one.centre;
//                    radius = one.radius;
//                }
//                else
//                {
//                    centre = two.centre;
//                    radius = two.radius;
//                }
//            }

//            // Otherwise we need to work with partially
//            // overlapping spheres
//            else
//            {
//                distance = Mathf.Sqrt((float)distance);
//                radius = (distance + one.radius + two.radius) * ((real)0.5);

//                // The new centre is based on one's centre, moved towards
//                // two's centre by an ammount proportional to the spheres'
//                // radii.
//                centre = one.centre;
//                if (distance > 0)
//                {
//                    centre += centreOffset * ((radius - one.radius) / distance);
//                }
//            }
//        }

//        /**
//         * Checks if the bounding sphere overlaps with the other given
//         * bounding sphere.
//         */
//        public bool overlaps(BoundingSphere other)
//        {
//            real distanceSquared = (centre - other.centre).SquareMagnitude();
//            return distanceSquared < (radius + other.radius) * (radius + other.radius);
//        }

//        /**
//         * Reports how much this bounding sphere would have to grow
//         * by to incorporate the given bounding sphere. Note that this
//         * calculation returns a value not in any particular units (i.e.
//         * its not a volume growth). In fact the best implementation
//         * takes into account the growth in surface area (after the
//         * Goldsmith-Salmon algorithm for tree construction).
//         */
//        public real getGrowth(BoundingSphere other)
//        {
//            BoundingSphere newSphere = new BoundingSphere(this, other);

//            // We return a value proportional to the change in surface
//            // area of the sphere.
//            return newSphere.radius * newSphere.radius - radius * radius;
//        }

//        /**
//         * Returns the volume of this bounding volume. This is used
//         * to calculate how to recurse into the bounding volume tree.
//         * For a bounding sphere it is a simple calculation.
//         */
//        public real getSize()
//        {
//            return ((real)1.333333) * Mathf.PI * radius * radius * radius;
//        }
//    };


//    /**
//    * Stores a potential contact to check later.
//    */
//    class PotentialContact
//    {
//        /**
//         * Holds the bodies that might be in contact.
//         */
//        public RigidBody[] body = new RigidBody[2];
//    };


//    /**
//     * A base class for nodes in a bounding volume hierarchy.
//     *
//     * This class uses a binary tree to store the bounding
//     * volumes.
//     */
//    class BVHNode<BoundingVolumeClass>
//    {

//        /**
//         * Holds the child nodes of this node.
//         */
//        public BVHNode<BoundingVolumeClass>[] children = new BVHNode<BoundingVolumeClass>[2];

//        /**
//         * Holds a single bounding volume encompassing all the
//         * descendents of this node.
//         */
//        public BoundingVolumeClass volume;

//        /**
//         * Holds the rigid body at this node of the hierarchy.
//         * Only leaf nodes can have a rigid body defined (see isLeaf).
//         * Note that it is possible to rewrite the algorithms in this
//         * class to handle objects at all levels of the hierarchy,
//         * but the code provided ignores this vector unless firstChild
//         * is null.
//         */
//        public RigidBody body;

//        // ... other BVHNode code as before ...

//        /**
//         * Holds the node immediately above us in the tree.
//         */
//        public BVHNode<BoundingVolumeClass> parent;

//        /**
//         * Creates a new node in the hierarchy with the given parameters.
//         */
//        public BVHNode(BVHNode<BoundingVolumeClass> parent, BoundingVolumeClass volume, RigidBody body=null)
//        {
//            this.children[0] = this.children[0] = null;
//            this.volume = volume;
//            this.body = body;
//            this.parent = parent;
//        }
//        //public BVHNode<BoundingVolumeClass>(BVHNode<BoundingVolumeClass> parent, BoundingVolumeClass<BoundingVolumeClass> volume,
//        //    RigidBody body = null)
//        //    : parent(parent), volume(volume), body(body)
//        //{
//        //    children[0] = children[1] = null;
//        //}

//        /**
//         * Checks if this node is at the bottom of the hierarchy.
//         */
//        public bool isLeaf()
//        {
//            return (body != null);
//        }

//        /**
//         * Checks the potential contacts from this node downwards in
//         * the hierarchy, writing them to the given array (up to the
//         * given limit). Returns the number of potential contacts it
//         * found.
//         */
//        public uint getPotentialContacts(PotentialContact[] contacts, uint limit)
//        {
//            // Early out if we don't have the room for contacts, or
//            // if we're a leaf node.
//            if (isLeaf() || limit == 0) return 0;

//            // Get the potential contacts of one of our children with
//            // the other
//            return children[0].getPotentialContactsWith(
//                children[1], contacts, limit
//                );
//        }

//        /**
//         * Inserts the given rigid body, with the given bounding volume,
//         * into the hierarchy. This may involve the creation of
//         * further bounding volume nodes.
//         */
//        public void insert(RigidBody newBody, BoundingVolumeClass newVolume)
//        {
//            // If we are a leaf, then the only option is to spawn two
//            // new children and place the new body in one.
//            if (isLeaf())
//            {
//                // Child one is a copy of us.
//                children[0] = new BVHNode<BoundingVolumeClass>(
//                    this, volume, body
//                    );

//                // Child two holds the new body
//                children[1] = new BVHNode<BoundingVolumeClass>(
//                    this, newVolume, newBody
//                    );

//                // And we now loose the body (we're no longer a leaf)
//                this.body = null;

//                // We need to recalculate our bounding volume
//                recalculateBoundingVolume();
//            }

//            // Otherwise we need to work out which child gets to keep
//            // the inserted body. We give it to whoever would grow the
//            // least to incorporate it.
//            else
//            {
//                if (children[0].volume.getGrowth(newVolume) <
//                    children[1].volume.getGrowth(newVolume))
//                {
//                    children[0].insert(newBody, newVolume);
//                }
//                else
//                {
//                    children[1].insert(newBody, newVolume);
//                }
//            }
//        }

//        /**
//         * Deltes this node, removing it first from the hierarchy, along
//         * with its associated
//         * rigid body and child nodes. This method deletes the node
//         * and all its children (but obviously not the rigid bodies). This
//         * also has the effect of deleting the sibling of this node, and
//         * changing the parent node so that it contains the data currently
//         * in that sibling. Finally it forces the hierarchy above the
//         * current node to reconsider its bounding volume.
//         */
//        ~BVHNode()
//        {
//            // If we don't have a parent, then we ignore the sibling
//            // processing
//            if (parent != null)
//            {
//                // Find our sibling
//                BVHNode<BoundingVolumeClass> sibling;
//                if (parent.children[0] == this) sibling = parent.children[1];
//                else sibling = parent.children[0];

//                // Write its data to our parent
//                parent.volume = sibling.volume;
//                parent.body = sibling.body;
//                parent.children[0] = sibling.children[0];
//                parent.children[1] = sibling.children[1];

//                // Delete the sibling (we blank its parent and
//                // children to avoid processing/deleting them)
//                sibling.parent = null;
//                sibling.body = null;
//                sibling.children[0] = null;
//                sibling.children[1] = null;
//                sibling = null;

//                // Recalculate the parent's bounding volume
//                parent.recalculateBoundingVolume();
//            }

//            // Delete our children (again we remove their
//            // parent data so we don't try to process their siblings
//            // as they are deleted).
//            if (children[0] != null)
//            {
//                children[0].parent = null;
//                children[0] = null;
//            }
//            if (children[1] != null)
//            {
//                children[1].parent = null;
//                children[1] = null;
//            }
//        }



//        /**
//         * Checks for overlapping between nodes in the hierarchy. Note
//         * that any bounding volume should have an overlaps method implemented
//         * that checks for overlapping with another object of its own type.
//         */
//        protected bool overlaps(BVHNode<BoundingVolumeClass> other)
//        {
//            return volume.overlaps(other.volume);
//        }

//        /**
//         * Checks the potential contacts between this node and the given
//         * other node, writing them to the given array (up to the
//         * given limit). Returns the number of potential contacts it
//         * found.
//         */
//        protected uint getPotentialContactsWith(BVHNode<BoundingVolumeClass> other,
//            PotentialContact[] contacts, uint limit)
//        {
//            // Early out if we don't overlap or if we have no room
//            // to report contacts
//            if (!overlaps(other) || limit == 0) return 0;

//            // If we're both at leaf nodes, then we have a potential contact
//            if (isLeaf() && other.isLeaf())
//            {
//                contacts[0].body[0] = body;
//                contacts[0].body[1] = other.body;
//                return 1;
//            }

//            // Determine which node to descend into. If either is
//            // a leaf, then we descend the other. If both are branches,
//            // then we use the one with the largest size.
//            if (other.isLeaf() ||
//                (!isLeaf() && volume.getSize() >= other.volume.getSize()))
//            {
//                // Recurse into ourself
//                uint count = children[0].getPotentialContactsWith(
//                    other, contacts, limit
//                    );

//                // Check we have enough slots to do the other side too
//                if (limit > count)
//                {
//                    return count + children[1].getPotentialContactsWith(
//                        other, contacts.Skip((int)count).ToArray(), limit - count
//                        );
//                }
//                else
//                {
//                    return count;
//                }
//            }
//            else
//            {
//                // Recurse into the other node
//                uint count = getPotentialContactsWith(
//                    other.children[0], contacts, limit
//                    );

//                // Check we have enough slots to do the other side too
//                if (limit > count)
//                {
//                    return count + getPotentialContactsWith(
//                        other.children[1], contacts.Skip((int)count).ToArray(), limit - count
//                        );
//                }
//                else
//                {
//                    return count;
//                }
//            }
//        }

//        /**
//     * For non-leaf nodes, this method recalculates the bounding volume
//     * based on the bounding volumes of its children.
//     */
//        protected void recalculateBoundingVolume(bool recurse = true)
//        {
//            if (isLeaf()) return;

//            // Use the bounding volume combining constructor.
//            volume = BoundingVolumeClass(
//                children[0].volume,
//                children[1].volume
//                );

//            // Recurse up the tree
//            if (parent!=null) parent.recalculateBoundingVolume(true);
//        }
//    };
//
//}
