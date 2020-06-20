public class HighSpeedTurn extends TrackFeature
{
	
	public HighSpeedTurn(int turnNo, TurnDirection direction, double distance, double roughness)
	{
		// Fill this method
		this.featureNo = turnNo;
		this.turnDirection = direction;
		this.distance = distance;
		this.roughness = roughness;
	}
	
	@Override
	public double getFeatureTypeMultiplier()
	{
		return 1.55;
	}
}
