import java.util.ArrayList;
import java.util.Collections;

public class Track
{
	
	private String trackName;
	private ArrayList<TrackFeature> featureList;
	private boolean isClockwise;
	
	private int currentFeature = 0;
	
	public Track()
	{
	}
	
	public Track(String trackName, ArrayList<TrackFeature> featureList, boolean isClockwise)
	{
		// Fill this method
		this.trackName = trackName;
		this.featureList = featureList;
		this.isClockwise = isClockwise;
	}
	
	public String getTrackName()
	{
		return trackName;
	}
	
	public void setTrackName(String trackName)
	{
		this.trackName = trackName;
	}
	
	public ArrayList<TrackFeature> getFeatureList()
	{
		return featureList;
	}
	
	public void setFeatureList(ArrayList<TrackFeature> featureList)
	{
		this.featureList = featureList;
	}
	
	public boolean isClockwise()
	{
		return isClockwise;
	}
	
	public void setClockwise(boolean clockwise)
	{
		isClockwise = clockwise;
	}
	
	public int getTrackLength()
	{
		// Fill this method
		int totalLength = 0;
		
		for (int i = 0; i < this.featureList.size(); i++)
		{
			totalLength += featureList.get(i).getDistance();
		}
		
		return totalLength;
	}
	
	public TrackFeature getNextFeature()
	{
		// Fill this method
		TrackFeature feature = featureList.get(currentFeature);
		currentFeature++;
		if (featureList.size() == currentFeature)
		{
			currentFeature = 0;
		}
		
		return feature;
	}
	
	public void addFeature(TrackFeature feature)
	{
		// Fill this method
		featureList.add(feature);
	}
	
	public boolean isValidTrack()
	{
		if (this.isClockwise())
		{
			Collections.sort(featureList, TrackFeature::sortByFeatureNoAscending);
		}
		else
		{
			Collections.sort(featureList, TrackFeature::sortByFeatureNoDescending);
		}
		
		
		if (featureList.get(0).turnDirection != TurnDirection.STRAIGHT ||
				featureList.get(featureList.size() - 1).turnDirection != TurnDirection.STRAIGHT)
		{
			return false;
		}
		
		int leftTurns = 0;
		int rightTurns = 0;
		
		for (int i = 0; i < featureList.size(); i++)
		{
			if (featureList.get(i).turnDirection == TurnDirection.LEFT)
			{
				leftTurns++;
			}
			else if (featureList.get(i).turnDirection == TurnDirection.RIGHT)
			{
				rightTurns++;
			}
		}
		
		if (this.isClockwise)
		{
			return rightTurns == leftTurns + 4;
		}
		else
		{
			return leftTurns == rightTurns + 4;
		}
	}
}
