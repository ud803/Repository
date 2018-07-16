import random

class Card_deck :

    def __init__(self):
        self.card_deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] *4
        self.cardDrawn = 0
        print("Card Deck Ready...")

    def Shuffle(self):
        random.shuffle(self.card_deck)
        print("Card Deck Shuffled...")

    def DrawCard(self):
        self.cardDrawn = self.card_deck.pop()
        return self.cardDrawn


class Dealer :

    def __init__(self):
        self.deck = []

    def Draw(self,drawCard):
        self.deck.append(drawCard)
        
    def showExceptOne(self):
        print(self.deck[:-1])

    def showAll(self):
        print("Show Dealer Deck")
        print(self.deck)
        
    def deckSum(self):
        return sum(self.deck)


class Player :

    def __init__(self):
        self.deck = []

    def Draw(self, drawCard):
        self.deck.append(drawCard)

    def showDeck(self):
        print(self.deck)

    def deckSum(self):
        return sum(self.deck)
    

class BlackJack :

    
    def __init__(self) :
        self.dealer = Dealer()
        self.player = Player()
        self.card = Card_deck()
        self.bet = 0
        self.p_status = "Neutral"  # "b" for blackjack, "bust" for bust, num for score
        self.d_status = "Neutral"  # "b" for blackjack, "bust" for bust, num for score 


    def evalPlayer(self):
        if(self.player.deckSum() == 21):
            self.p_status = 'b'
        elif(self.player.deckSum() > 21):
            self.p_status = 'bust'
        elif(self.player.deckSum() < 21) :
            self.p_status = self.player.deckSum()

        
    def evalDealer(self):
        if(self.dealer.deckSum() == 21):
            self.d_status = 'b'
        elif(self.dealer.deckSum() > 21):
            self.d_status = 'bust'
        else :
            self.d_status = self.dealer.deckSum()


    def playerDraw(self):
        self.player.Draw(self.card.DrawCard())
        self.evalPlayer()
            

    def dealerDraw(self):
        self.dealer.Draw(self.card.DrawCard())
        self.evalDealer()

     
    def cardShuffle(self) :
        print("Welcome to the Yonsei BlackJack")
        bet = input("How much to bet?")
        self.card.Shuffle()
        self.dealer = Dealer()
        self.player = Player()
        print("Check your cards")

        for i in range(2):
            self.player.Draw(self.card.DrawCard())
            self.dealer.Draw(self.card.DrawCard())


    def playerCase(self):
        self.evalPlayer()

        while(self.p_status not in ['b', 'bust']):
            hitorstay = input("Hit or Stay?(1/0)")
            if(hitorstay == "1") :
                self.playerDraw()
                self.player.showDeck()
            else :
                break

            
    def dealerCase(self):
        while(self.dealer.deckSum() < 16):
            print("Dealer lower than 16. Dealer Draws.")
            self.dealerDraw()
            self.dealer.showAll()
        if( self.p_status != 'bust'):
            if( type(self.d_status) == int and type(self.p_status)==int):
                if( self.d_status < self.p_status):
                    if( 16 <= self.dealer.deckSum() < 20) :
                        if(random.randint(0,10) >= 7) :
                            print("Dealer decides to draw...")
                            self.dealerDraw()
                            self.dealer.showAll()
                
                    elif ( self.dealer.deckSum() == 20) :
                        if(random.randint(0,10) >= 9) :
                            print("Dealer decides to draw!!!")
                            self.dealerDraw()
                            self.dealer.showAll()   


    def compResult(self):
        if( self.p_status == 'b'):
            if(self.d_status == 'b') :
                print("Both you and dealer got blackjack! You get {0} dollars!".format(self.bet))
            else :
                print("You got blackjack! You get {0} dollars!".format(self.bet*2))
        elif( self.p_status == 'bust') :
            print("Bust! You lost all your money!")
        else :
            if( self.d_status == 'bust') :
                print("You won! You get {0} dollars!".format(self.bet*2))
            elif( self.player.deckSum() > self.dealer.deckSum()) :
                print("You won! You get {0} dollars!".format(self.bet*2))
            elif (self.player.deckSum() < self.dealer.deckSum()):
                print("You Lost! You lost all your money!")
            else :
                print("Tie!")

        print("Player sum: ", self.player.deckSum())
        print("Dealer sum: ", self.dealer.deckSum())

    
    def GameBegins(self):

        trig = True

        while(trig == True) :
            self.cardShuffle()
            self.player.showDeck()
            self.dealer.showExceptOne()
            
            self.playerCase()
            self.dealer.showAll()
            self.dealerCase()
            
            self.compResult()
            
            play_again = input("Play Again?(1/0)")
            if(play_again == "0"):
                trig = False

                
blackjack = BlackJack()                                    
blackjack.GameBegins()
