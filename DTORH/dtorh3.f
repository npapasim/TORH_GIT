         SUBROUTINE DTORH3(Z,MDIM,NDIM,MMAX,NMAX,PL,QL,NEWM,NEWN)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  INPUT :                                                         C         
C    Z        ARGUMENT OF THE FUNCTIONS                            C       
C    MDIM     M-DIMENSION OF THE ARRAYS. MDIM MUST BE GREATER      C
C             THAN MMAX                                            C                    
C    NDIM     N-DIMENSION OF THE ARRAYS. NDIM MUST BE GREATER      C
C             THAN NMAX                                            C                                
C    MMAX     MAXIMUM ORDER  OF THE FUNCTIONS :                    C                  
C              WE CALCULATE FUNCTIONS OF ALL ORDERS BELOW MMAX.    C                   
C    NMAX     MAXIMUM DEGREE  OF THE FUNCTIONS :                   C                  
C              WE CALCULATE FUNCTIONS OF ALL DEGREES BELOW         C       
C              MIN(NMAX,NEWN).                                     C                                                                                C
C  OUTPUT :                                                        C       
C   *IF MODE IS EQUAL TO 0:                                        C                                                                               
C    PL(M,N)                                                       C       
C             THESE VALUES ARE KEPT IN AN ARRAY                    C       
C    QL(M,N)                                                       C       
C             THESE VALUES ARE KEPT IN AN ARRAY                    C       
C                                                                  C       
C    NEWM     MAXIMUM  ORDER OF FUNCTIONS CALCULATED WHEN          C       
C             QL (MMAX,0)   IS LARGER THAN 1/TINY                  C       
C              (OVERFLOW LIMIT = 1/TINY, TINY IS DEFINED BELOW).   C       
C    NEWN     MAXIMUM  DEGREE OF FUNCTIONS CALCULATED WHEN         C                 
C             PL (M,NMAX)   IS LARGER THAN 1/TINY  FOR SOME        C       
C             M=0,...,NEWM                                         C       
C              (OVERFLOW LIMIT = 1/TINY, TINY IS DEFINED BELOW).   C       
C    NOTE1: FOR A PRECISION OF 10**(-12), IF Z>5 AND (Z/M)>0.22    C
C           THE CODE USES A SERIES EXPANSION FOR PL(M,0). WHEN     C
C           Z<20 AND (Z/M)<0.22 A CONTINUED FRACTION IS APPLIED.   C                           
C    NOTE2: FOR A PRECISION OF 10**(-8), IF Z>5 AND (Z/M)>0.12     C
C           THE CODE USES A SERIES EXPANSION FOR PL(M,0). WHEN     C
C           Z<20 AND (Z/M)<0.12 A CONTINUED FRACTION IS APPLIED.   C                                                                                                    
C   *IF MODE IS EQUAL TO 1:                                        C       
C      THE SET OF FUNCTIONS EVALUATED IS:                          C       
C                 PL(M,N)/GAMMA(M+1/2),QL(M,N)/GAMMA(M+1/2),       C       
C      WHICH ARE RESPECTIVELY STORED IN THE ARRAYS PL(M,N),QL(M,N) C       
C      NEWM AND NEWN REFER TO THIS NEW SET OF FUNCTIONS            C       
C      NOTE1 AND NOTE2 ALSO APPLY IN THIS CASE                     C       
C   *IF MODE IS EQUAL TO 2:                                        C       
C      THE CODE PERFORMS AS FOR MODE 1, BUT THE RESTRICTION Z<20   C
C      FOR THE EVALUATION OF THE CONTINUED FRACTION IS NOT         C
C      CONSIDERED                                                  C
C      WARNING: USE ONLY IF HIGH M'S FOR Z>20 ARE REQUIRED. THE    C      
C      EVALUATION OF THE CF MAY FAIL TO CONVERGE FOR TOO HIGH Z'S  C      
C  PARAMETERS:                                                     C       
C   MODE: ENABLES THE ENLARGEMENT OF THE RANGE OF ORDERS AND       C
C         DEGREES THAT CAN BE EVALUATED.                           C                    
C   EPS:  CONTROLS THE ACCURACY OF THE CONTINUED FRACTIONS         C
C         AND SERIES.                                              C
C   IPRE: REQUIRED PRECISION IN THE EVALUATION OF TOROIDAL         C
C         HARMONICS.                                               C
C           *IF IPRE=1, PRECISION=10**(-12) (TAKING EPS<10**(-12)) C                                                                                C
C           *IF IPRE=2, PRECISION=10**(-8) (TAKING EPS<10**(-8))   C       
C   TINY: SMALL PARAMETER NEAR THE UNDERFLOW LIMIT.                C                   
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C   DECLARATION OF VARIABLES    C                                                                                                                  C                                                                         C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
       INTEGER MDIM,NDIM,MMAX,NMAX,NEWM,NEWN,MP,ICAL,M,N,NP,I,
     *   MODE,IPRE,L  
       DOUBLE PRECISION Z,PI,EPS,TINY,OVER,TINYSQ,QZ,QDC1,
     *   QARGU,ARGU1,PISQ,D1,D2,FL,CC,AR,GAMMA,DFACQS,FCP,
     *   DD,PL0,QM0,DFAC3,FC,DFACC,FC2,DFAC4,GAMMAH,ELLIP1,
     *   ELLIP2,FACTCO
       DOUBLE PRECISION PL(0:MDIM,0:NDIM),QL(0:MDIM,0:NDIM),PR(2)
       PARAMETER(PI=3.14159265358979323D0,EPS=1.D-14,TINY=1.D-290,
     * MODE=0,IPRE=1)
       OVER=1.D0/TINY
       TINYSQ=DSQRT(TINY)
       IF ((IPRE.NE.1).AND.(IPRE.NE.2)) THEN
         WRITE(6,*)'IPRE MUST BE 1 OR 2'
         STOP
       END IF
       PR(1)=.22D0
       PR(2)=.12D0
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C   EPS: REQUIRED ACCURACY FOR THE CONTINUED FRACTION     C
C        (MODIFIED LENTZ)                                 C
C   TINY: SMALL PARAMETER TO PREVENT OVERFLOWS IN THE CF  C
C         (CLOSE TO THE UNDERFLOW LIMIT)                  C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
       IF (Z.LE.1.D0) THEN
          WRITE(6,*)'IMPROPER ARGUMENT. Z MUST BE GREATER THAN 1'
          STOP
       END IF
       QZ=Z
       QDC1=QZ*QZ-1.D0
       QARGU=QZ/DSQRT(QDC1)
       ARGU1=DSQRT(2.D0/(Z+1.D0))
       PISQ=DSQRT(PI)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CC   WE EVALUATE Q^{0}_{-1/2},Q^{1}_{-1/2}            C
CC   USING SLATEC ROUTINES FOR ELLIPTIC FUNCTIONS     C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
       QL(0,0)=ARGU1*ELLIP1(ARGU1)
       QL(1,0)=-1.D0/DSQRT(2.D0*(QZ-1.D0))
     *          *ELLIP2(ARGU1)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CC    WE APPLY FORWARD RECURRENCE IN M FOR Q'S        C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
        MP=1
         IF (MODE.EQ.0) THEN
1          IF ((MP.LE.MMAX).AND.(ABS(QL(MP,0)).LT.OVER)) THEN
             QL(MP+1,0)=-2.D0*MP*QARGU*QL(MP,0) 
     *       -(MP-0.5D0)*(MP-0.5D0)*QL(MP-1,0) 
             MP=MP+1
             GOTO 1
           ENDIF    
           IF ((MP-1).LT.MMAX) MMAX=MP-1
           NEWM=MMAX
        ELSE
           QL(0,0)=QL(0,0)/PISQ
           QL(1,0)=QL(1,0)*2.D0/PISQ
2          IF ((MP.LE.MMAX).AND.(ABS(QL(MP,0)).LT.OVER)) THEN
             D1=MP+0.5D0
             QL(MP+1,0)=-2.D0*MP*QARGU*QL(MP,0)/D1
     *       -(MP-0.5D0)*QL(MP-1,0)/D1
             MP=MP+1
             GOTO 2
           ENDIF
           IF ((MP-1).LT.MMAX) MMAX=MP-1
           NEWM=MMAX
        END IF
        FL=MMAX/2.D0
        CC=ABS(FLOAT(INT(FL))-FL)
         IF (CC.LT.0.4D0) THEN
           AR=1.
          ELSE
           AR=-1.
         END IF
       IF (MODE.EQ.0) THEN
          GAMMA=GAMMAH(MMAX,OVER)*AR*PISQ  
          IF (ABS(GAMMA).LT.TINY) THEN
             WRITE(6,*)'MMAX IS TOO LARGE FOR MODE=0'
             WRITE(6,*)'BETTER TRY MODE=1'
             STOP
          END IF              
        ELSE 
          GAMMA=AR
        END IF
        DFACQS=-(GAMMA/QL(MMAX+1,0))*GAMMA/PI
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CC  EVALUATION OF PL(MMAX,0),PL(MMAX+1,0)           C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CC  WE CHOOSE EXPANSION OR CF FOR PL(MMAX,0)        C
CC  DEPENDING ON THE VALUES OF Z,MMAX AND MODE      C                           
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
       ICAL=1
       IF ((Z/MMAX).GT.PR(IPRE)) ICAL=2
       IF (Z.LT.5.D0) THEN
         ICAL=1
       ELSE IF (Z.GT.20.D0) THEN
         IF ((MODE.NE.2).AND.(ICAL.EQ.1)) ICAL=0
       END IF
       IF (ICAL.EQ.0) THEN
         WRITE(6,*)'YOU MUST CHOOSE MODE=2'
         STOP
       END IF
        IF (ICAL.EQ.1) THEN
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC        
CC   WE CALCULATE THE CF FOR P'S    C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
           CALL FRACPS(QARGU,MMAX+1,0,EPS,TINYSQ,FCP) 
           DD=MMAX+0.5D0
           IF (MODE.NE.0) THEN
            FCP=FCP/DD
            DFACQS=DFACQS/DD
           END IF
           PL(MMAX,0)=DFACQS/DSQRT(QDC1)
     *        /(1.D0-FCP*QL(MMAX,0)/QL(MMAX+1,0))
           PL(MMAX+1,0)=FCP*PL(MMAX,0)
        ELSE
           CALL EXPAN(Z,MODE,IPRE,OVER,QARGU,MMAX,PL0)
           PL(MMAX,0)=PL0
           DD=MMAX+0.5D0
           IF (MODE.NE.0) THEN
            FCP=FCP/DD
            DFACQS=DFACQS/DD
           END IF
           PL(MMAX+1,0)=(QL(MMAX+1,0)/QL(MMAX,0))*
     *         (PL(MMAX,0)- DFACQS/DSQRT(QDC1))
        END IF
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CC   EVALUATION OF PL(MMAX,1)    C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
        QM0=QL(MMAX,0)
        DFAC3=(GAMMA/QM0)*GAMMA/PI/(0.5D0-MMAX)
        CALL FRAC(Z,MMAX,0,EPS,TINYSQ,FC)
        PL(MMAX,1)=PL(MMAX,0)*FC+DFAC3
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CC   EVALUATION OF PL(MMAX+1,1)  C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
        QM0=QL(MMAX+1,0)
        IF (MODE.EQ.0) THEN 
           DFACC=(0.5D0+MMAX)
        ELSE
           DFACC=1.D0/(0.5D0+MMAX)
        END IF      
        DFAC3=-(GAMMA/QM0)*GAMMA*DFACC/PI
        CALL FRAC(Z,MMAX+1,0,EPS,TINYSQ,FC2)
        PL(MMAX+1,1)=PL(MMAX+1,0)*FC2+DFAC3
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CC  WE APPLY BACKWARD RECURRENCE OVER M TO GET THE  C
CC  SET PL(M,0),PL(M,1)                             C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
       IF (MODE.EQ.0) THEN
         DO 3 I=1,MMAX
           MP=MMAX+1-I
           M=MP-1
           PL(M,0)=-(PL(MP+1,0)+2.D0*MP*QARGU*PL(MP,0))
     *              /((0.5D0-MP)*(0.5D0-MP))
3        CONTINUE
         DO 4 I=1,MMAX
           MP=MMAX+1-I
           M=MP-1
           PL(M,1)=(PL(MP+1,1)+2.D0*MP*QARGU*PL(MP,1))
     *              /((1.5D0-MP)*(0.5D0+MP))
4        CONTINUE
       ELSE
         DO 5 I=1,MMAX
           MP=MMAX+1-I
           M=MP-1
           PL(M,0)=((MP+0.5D0)*PL(MP+1,0)+2.D0*MP*QARGU*PL(MP,0))
     *              /(0.5D0-MP)
5        CONTINUE
         DO 6 I=1,MMAX
           MP=MMAX+1-I
           M=MP-1
           PL(M,1)=((MP+0.5D0)*PL(MP+1,1)+2.D0*MP*QARGU*PL(MP,1))*
     *              (MP-0.5D0)/((1.5D0-MP)*(0.5D0+MP))
6        CONTINUE
       END IF
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CC  NOW, WE PERFORM THE EVALUATION OVER N FOR EACH M  C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CC      WE START EVALUATING THE SET OF PL(M,N)        C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
       DO 7 L=0,MMAX
         M=L
         FL=M/2.D0
         CC=ABS(FLOAT(INT(FL))-FL)
         IF (CC.LT.0.4D0) THEN
           AR=1.
          ELSE
           AR=-1.
         END IF
         IF (MODE.EQ.0) THEN
           GAMMA=GAMMAH(M,OVER)*AR*PISQ
         ELSE 
          GAMMA=AR
         END IF
         NP=1
15       IF ((NP.LE.NMAX).AND.
     *    (ABS((NP-M+0.5D0)*PL(L,NP)).LT.OVER)) THEN
              PL(L,NP+1)=(2.D0*NP*Z*PL(L,NP)
     *         -(NP+M-0.5D0)*PL(L,NP-1))/(NP-M+0.5D0)
              NP=NP+1
              GOTO 15
         ENDIF
         NMAX=NP-1
         NEWN=NMAX
7      CONTINUE
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CC      WE EVALUATE THE SET OF QL(M,N)        C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC       
       DO 8 L=0,MMAX
         M=L
         FL=M/2.D0
         CC=ABS(FLOAT(INT(FL))-FL)
         IF (CC.LT.0.4D0) THEN
           AR=1.
          ELSE
           AR=-1.
         END IF
         IF (MODE.EQ.0) THEN
           GAMMA=GAMMAH(M,OVER)*AR*PISQ
         ELSE 
          GAMMA=AR
         END IF     
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CC   WE EVALUATE THE C.F. FOR Q'S USING LENTZ-THOMPSON        C
CC   IF M=0,1. IN OTHER CASE WE CALCULATE QL(M,NMAX) AND      C
CC   QL(M,NMAX+1) FROM THE RECURRENCE.                        C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
       IF (M.EQ.2) THEN
         IF (MODE.EQ.0) THEN
           MP=1
9          IF (MP.LE.MMAX) THEN
            QL(MP+1,NMAX)=-2.D0*MP*QARGU*QL(MP,NMAX) 
     *       -(MP-NMAX-0.5D0)*(MP+NMAX-0.5D0)*QL(MP-1,NMAX) 
             MP=MP+1
             GOTO 9
           ENDIF
           MP=1
10         IF (MP.LE.MMAX) THEN
            QL(MP+1,NMAX+1)=-2.D0*MP*QARGU*QL(MP,NMAX+1) 
     *       -(MP-NMAX-1.5D0)*(MP+NMAX+0.5D0)*QL(MP-1,NMAX+1) 
             MP=MP+1
             GOTO 10
           ENDIF
        ELSE
           MP=1
11         IF (MP.LE.MMAX) THEN
            D1=MP+0.5D0
            D2=MP-0.5D0
            QL(MP+1,NMAX)=-2.D0*MP*QARGU*QL(MP,NMAX)/D1
     *      -(MP-NMAX-0.5D0)*(MP+NMAX-0.5D0)*QL(MP-1,NMAX)/(D1*D2)
            MP=MP+1
            GOTO 11
           ENDIF
           MP=1
12         IF (MP.LE.MMAX) THEN
            D1=MP+0.5D0
            D2=MP-0.5D0
            QL(MP+1,NMAX+1)=-2.D0*MP*QARGU*QL(MP,NMAX+1)/D1 
     *      -(MP-NMAX-1.5D0)*(MP+NMAX+0.5D0)
     *      *QL(MP-1,NMAX+1)/(D1*D2)
             MP=MP+1
             GOTO 12
           ENDIF
        END IF
       END IF
       IF ((M.EQ.0).OR.(M.EQ.1)) THEN
        CALL FRAC(Z,M,NMAX,EPS,TINYSQ,FC)
        DFAC4=(FACTCO(NMAX,PL(L,NMAX+1),M)*GAMMA)
     *        *GAMMA/PI/(NMAX+M+0.5D0)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC 
CC     EVALUATION OF QL(L,NMAX+1) AND QL(L,NMAX) USING        C
CC     THE WRONSKIAN W{PL(L,NMAX),QL(L,NMAX)},                C
CC     THE KNOWN VALUES OF PL(L,NMAX+1) AND PL(L,NMAX)        C
CC     THE VALUE OF H = QL(L,NMAX+1)/QL(L,NMAX)               C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
        QL(L,NMAX)=DFAC4/(1.D0-FC*PL(L,NMAX)/PL(L,NMAX+1))
        QL(L,NMAX+1)=QL(L,NMAX)*FC
       END IF       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CC   WE USE THE BACKWARD RECURRENCE RELATION FOR Q'S          C                                                      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
         DO 13 I=1,NMAX
           NP=NMAX+1-I
           N=NP-1
           QL(L,N)=((NP+NP)*Z*QL(L,NP)-(NP-M+0.5D0)
     *      *QL(L,NP+1))/(NP+M-0.5D0)
13       CONTINUE
8       CONTINUE
        RETURN
        END
                     
