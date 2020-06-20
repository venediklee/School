import java.lang.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;


class Main
{
	public static void main(String[] args)
	{
		Scanner input = new Scanner(System.in);
		
		String gridSize = input.nextLine();
		List<Integer> rowAndColumnCount = Arrays.stream(gridSize.split(" ")).map(Integer::parseInt).collect(Collectors.toList());
		int numberOfUsers = Integer.parseInt(input.nextLine());
		
		List<String> userAndWantedSeatNames = new ArrayList<>(numberOfUsers);
		for (int i = 0; i < numberOfUsers; i++) // get users and seats they want to reserve
		{
			userAndWantedSeatNames.add(input.nextLine());
		}
		
		//create seats
		ArrayList<ArrayList<Seat>> seats = new ArrayList<>(rowAndColumnCount.get(0));
		for (int i = 0; i < rowAndColumnCount.get(0); i++)
		{
			seats.add(new ArrayList<>(rowAndColumnCount.get(1)));
		}
		
		for (char row = 'A'; row < 'A' + rowAndColumnCount.get(0); row++)
		{
			for (int column = 0; column < rowAndColumnCount.get(1); column++)
			{
				seats.get(row - 'A').add(new Seat(row + Integer.toString(column)));
			}
		}
		
		Logger.InitLogger();//start logging
		List<Thread> userThread = new ArrayList<Thread>(numberOfUsers);
		for (int i = 0; i < numberOfUsers; i++)
		{
			User user = new User(Arrays.asList(userAndWantedSeatNames.get(i).split(" ")), seats, rowAndColumnCount.get(1));
			userThread.add(new Thread(user));
			userThread.get(i).start();
		}
		
		//wait for threads
		for (int i = 0; i < numberOfUsers; i++)
		{
			try
			{
				userThread.get(i).join();
			}
			catch (InterruptedException e)
			{
				//System.out.println("Exception" + e + "is caught in when joining user thread");
			}
		}
		
		//print final table
		for (char row = 'A'; row < 'A' + rowAndColumnCount.get(0); row++)
		{
			for (int column = 0; column < rowAndColumnCount.get(1); column++)
			{
				Seat seat = seats.get(row - 'A').get(column);
				if (seat.NameOfReserver == null)
				{
					System.out.print("E: ");
				}
				else
				{
					System.out.print("T:" + seat.NameOfReserver + " ");
				}
			}
			System.out.println();
		}
	}
}