import java.awt.*;
import java.util.Comparator;

public abstract class TrackFeature
{
	
	protected int featureNo;
	protected TurnDirection turnDirection;
	protected double distance;
	protected double roughness;
	
	public int getFeatureNo()
	{
		return featureNo;
	}
	
	public double getRoughness()
	{
		return roughness;
	}
	
	public double getDistance()
	{
		return distance;
	}
	
	public TurnDirection getTurnDirection()
	{
		return turnDirection;
	}
	
	public int sortByFeatureNoAscending(TrackFeature another)
	{
		if (this.getFeatureNo() < another.getFeatureNo())
		{
			return -1;
		}
		else
		{
			return 1;
		}
	}
	
	public int sortByFeatureNoDescending(TrackFeature another)
	{
		if (this.getFeatureNo() < another.getFeatureNo())
		{
			return 1;
		}
		else
		{
			return -1;
		}
	}
	
	public abstract double getFeatureTypeMultiplier();
	
}
