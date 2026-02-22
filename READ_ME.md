# HOW TO START PROJECT: 
1.In three individual terminals, run the commands IN THIS ORDER:
<ol>
  <li> python reversi_server.py </li>
  <li> python greedy_player.py </li>
  <li> python greedy_player.py (For second player) </li>
</ol>

The Runner game window will display after starting within this order where a game of reversi will commence until the game is completed with one player dominating over the other. This will oepn a pygame display window the will show you the game and its pieces run through the animation. 

Once the window has displayed, `click` on the window to have the game animation start and let it run through until the game has stoped and neither hand can make any more moves. To close the window, simply `click` on the window again to close out the window. 

NOTE: in the seperate terminals for the players, `1` represets the white hand, `-1` represents the black hand.

# REVERSI RULES:
(WATCH THIS VIDEO, this will cover the basics of REVERSI RULES)
https://www.youtube.com/watch?v=4XdyAZhzJW8

1. Reversi is a two-player strategy game played on an 8x8 board using discs that are colored white on one side and black on the other. One player plays the discs black side up while his opponent plays the discs white side up.

2. The object of the game is to place your discs on the board so as to outflank your opponent's discs, flipping them over to your color. The player who has the most discs on the board at the end of the game wins.

3. board positions are denoted by a letter representing the column (A through H) and a number representing the row (1 through 8). For example, the top-left square on the board is referred to as A1 while the square to the right of it is referred to as B1.

4. Every game starts with four discs placed in the center of the board

5. Players take turns making moves. A move consists of a player placing a disc of his color on the board. The disc must be placed so as to outflank one or more opponent discs, which are then flipped over to the current player's color.

6. Outflanking your opponent means to place your disc such that it traps one or more of your opponent's discs between another disc of your color along a horizontal, vertical or diagonal line through the board square

7. If a player cannot make a legal move, he forfeits his turn and the other player moves again (this is also known as passing a turn). Note that a player may not forfeit his turn voluntarily. If a player can make a legal move on his turn, he must do so.

8. The game ends when neither player can make a legal move. This includes when there are no more empty squares on the board or if one player has flipped over all of his opponent's discs (a situation commonly known as a wipeout)

9. The player with the most discs of his color on the board at the end of the game wins. The game is a draw if both players have the same number of discs
(NOTE: When making a move, you may outflank your opponent's discs in more than one direction. All outflanked discs are flipped)

<ul>
SEE HERE FOR REFERENCE: https://documentation.help/Reversi-Rules/rules.htm 

FOR REFERENCE ON STRATEGY: https://documentation.help/Reversi-Rules/strategy.htm 
</ul>

