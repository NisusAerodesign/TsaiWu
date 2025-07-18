import numpy as np
import matplotlib.pyplot as plt


class Tsai_Wu:
    """
    Implementação do critério de falha de Tsai-Wu.
    Seguindo a teoria presente em:
    DANIEL, I. M.; ORI ISHAI. Engineering mechanics of composite materials. Nova York, EUA: Oxford University Press, 2006.
    Implementado por Vinicius Miranda Rodrigues
    contato: (41)99987-3264 ou vini142@hotmail.com
    """

    def __init__ (self, Xc, Xt, Zc, Zt, Sxy, Syz, Yc = None, Yt = None, Sxz = None):
        """
        Inicializando o objeto
        
        Parâmetros
        ----------
        Xc : float
            Tensão última de compressão no eixo x
        Xt : float
            Tensão última de tração no eixo x
        Zc : float
            Tensão última de compressão no eixo z
        Zt : float
            Tensão última de tração no eixo z
        Sxy : float
            Tensão última de cisalhamento no plano xy
        Syz : float
            Tensão última de cisalhamento no plano yz
        Yc : float
            Tensão última de compressão no eixo y
        Yt : float
            Tensão última de tração no eixo y
        Sxz : float
            Tensão última de cisalhamento no plano xz
        """
        self.F11 = 1/Xc/Xt
        self.F33 = 1/Zc/Zt
        self.F1  = (Xc-Xt)/Xc/Xt
        self.F3  = (Zc-Zt)/Zc/Zt
        self.F66 = 1/(Sxy**2)
        self.F44 = 1/(Syz**2)
        if Yc is not None and Yt is not None and Sxz is not None:
            self.F2  = (Yc-Yt)/Yc/Yt
            self.F22 = 1/Yc/Yt
            self.F55 = 1/(Sxz**2)
        else:
            self.F2  = self.F1
            self.F22 = self.F11
            self.F55 = self.F44
        self.F12 = 4*np.sqrt(self.F66) - (2*self.F1*np.sqrt(self.F66) + 2*self.F2*np.sqrt(self.F66) + self.F11 +self.F22 + self.F66)
        self.F13 = 4*np.sqrt(self.F55) - (2*self.F1*np.sqrt(self.F55) + 2*self.F3*np.sqrt(self.F55) + self.F11 +self.F33 + self.F55)
        self.F23 = 4*np.sqrt(self.F44) - (2*self.F2*np.sqrt(self.F44) + 2*self.F3*np.sqrt(self.F44) + self.F22 +self.F33 + self.F44)
        #self.F12 = self.F13 = self.F23 = -1 #parece ser o utilizado pelo ansys, mas é estranho pela teoria
    
    def criterio(self, x = 0, y = 0, z = 0, xy = 0, xz = 0, yz = 0):
        """
        Recebe as cargas a serem analisadas e calcula o coeficiente de segurança, também printa se ele falha ou não.
        
        Parâmetros
        ----------
        x  : float
            Tensão em x, + => tração e - => compressão
        y  : float
            Tensão em y, + => tração e - => compressão
        z  : float
            Tensão em z, + => tração e - => compressão
        xy : float
            Tensão de cisalhamento em xy
        xz : float
            Tensão de cisalhamento em xz
        yz : float
            Tensão de cisalhamento em yz
        
        Retorna
        ----------
        self : Tsai_Wu
            Retorna self para uso intuitivo do código
        """
        R = self.F1*x + self.F2*y + self.F3*z + self.F11*(x**2) + self.F22*(y**2) + self.F33*(z**2) + self.F44*(yz**2) + self.F55*(xz**2) + self.F66*(xy**2) + 2*self.F12*x*y + 2*self.F13*x*z + 2*self.F23*y*z
        if np.abs(R) >= 1:
            A = self.F11*x**2 + self.F22*y**2 + self.F33*z**2 + self.F44*yz + self.F55*xz + self.F66*xy - self.F12*x*y - self.F13*x*z - self.F23*y*z
            B = self.F1*x +self.F2*y + self.F3*z
            print(f"O material vai falhar, R = {R}, fator de segurança = {1/2/A*(np.sqrt((B**2) + 4*A) - B)}") # bhaskara para encontrar o fator pelo qual deveria ser multiplicado todas as tensões para alcançar a falha
        else:
            A = self.F11*x**2 + self.F22*y**2 + self.F33*z**2 + self.F44*yz + self.F55*xz + self.F66*xy - self.F12*x*y - self.F13*x*z - self.F23*y*z
            B = self.F1*x +self.F2*y + self.F3*z
            print(f"O material não vai falhar, R = {R}, fator de segurança = {1/2/A*(np.sqrt((B**2) + 4*A) - B)}") # bhaskara para encontrar o fator pelo qual deveria ser multiplicado todas as tensões para alcançar a falha
        return self

def tensao(h, b, t, MF, MTp, FC, MTt):
    """
    Calcula as tensões em determinada seção transversal automaticamente, para caixa vazada como é feito na planilha, a partir das dimensões e forças.

    Parâmetros
    ----------
        h   : float
            Altura da caixa
        b   : float
            Base da caixa
        t   : float
            E
            Espessura das paredes
        MF  : float
            Momento fletor (Distribuição de sustentação da asa da ponta até a seção avaliada)
        MTp : float
            Momento torçor de perfil (Momento gerado pelo perfil da asa)
        FC  : float
            Força cortante (Sustentação da asa no ponto da seção avaliada)
        MTt : float
            Momento torçor do tailboom (Momento gerado pela força do EH)

    Retorna
    ----------
        X  : float
            Tensão no eixo x
        yz : float
            Tensão de cisalhamento  no plano yz
        0  : int
            Retorna zeros em posições especificas para permitir que o output seja colocado diretamente no input do critério de falha
    """
    Ix = b*(h**3)/12 - ((b - 2*t)*((h - 2*t)**3))/12
    Iy = h*(b**3)/12 - ((h - 2*t)*((b - 2*t)**3))/12
    J = Ix + Iy
    Q = (h*b - (h - 2*t)*(b - 2*t))*h/2
    X = -(MF*(h/2)/Ix)
    yz = np.abs(MTt*(h/2)/J) + np.abs(MTp*(h/2)/J) + np.abs((FC*Q)/(Ix*(2*t)))
    return X, 0, 0, 0, 0, yz

if __name__ == '__main__':
    Tsai = Tsai_Wu(4.206e8, 5.629e8, 1.444e8, 4.938e7, 4.81e7, 2.203e6)
    Tsai.criterio(*tensao(20e-3, 2e-3, 1.2e-3, 100000e-3, 0, -1000, 0))
    #Tsai.criterio(55.55e6,55.55e6,0,0,0,0) verificar