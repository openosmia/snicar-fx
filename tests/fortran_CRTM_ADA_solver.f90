MODULE CRTM_Shared
  IMPLICIT NONE

  INTEGER, PARAMETER :: fp = KIND(1.0D0)

  ! -----------------
  ! Literal constants
  ! -----------------
  REAL(fp), PUBLIC, PARAMETER :: ZERO         = 0.0_fp
  REAL(fp), PUBLIC, PARAMETER :: ONE          = 1.0_fp
  REAL(fp), PUBLIC, PARAMETER :: TWO          = 2.0_fp
  REAL(fp), PUBLIC, PARAMETER :: THREE        = 3.0_fp
  REAL(fp), PUBLIC, PARAMETER :: FOUR         = 4.0_fp
  REAL(fp), PUBLIC, PARAMETER :: FIVE         = 5.0_fp
  REAL(fp), PUBLIC, PARAMETER :: TEN          = 10.0_fp
  REAL(fp), PUBLIC, PARAMETER :: POINT_25     = 0.25_fp
  REAL(fp), PUBLIC, PARAMETER :: POINT_5      = 0.5_fp
  REAL(fp), PUBLIC, PARAMETER :: POINT_75     = 0.75_fp
  REAL(fp), PUBLIC, PARAMETER :: ONEpointFIVE = 1.5_fp
  
  ! --------------------
  ! PI-related constants
  ! --------------------
  REAL(fp), PUBLIC, PARAMETER :: PI = 3.141592653589793238462643383279_fp
  REAL(fp), PUBLIC, PARAMETER :: TWOPI = TWO * PI
  REAL(fp), PUBLIC, PARAMETER :: DEGREES_TO_RADIANS = PI / 180.0_fp
  REAL(fp), PUBLIC, PARAMETER :: RADIANS_TO_DEGREES = 180.0_fp / PI

  ! Constants
  REAL(fp), PARAMETER :: DELTA_OPTICAL_DEPTH = 1.0e-8_fp
  REAL(fp), PARAMETER :: MAX_ALBEDO = 0.999999_fp
  REAL(fp), PUBLIC, PARAMETER :: SCATTERING_ALBEDO_THRESHOLD = 1.0e-10_fp
  INTEGER, PARAMETER :: SUCCESS = 0
  INTEGER, PARAMETER :: FAILURE = 1

  ! Grid sizes
  INTEGER, PARAMETER :: nL = 2   ! number of layers
  INTEGER, PARAMETER :: nA = 6   ! number of angles
  INTEGER, PARAMETER :: nS = 4   ! number of streams

  ! Global status and flags
  INTEGER :: Error_Status = 0, mth_Azi = 0
  LOGICAL :: Solar_Flag_true = .TRUE.

  ! Common radiative transfer scalars
  REAL(fp) :: COS_SUN = ZERO
  REAL(fp) :: Solar_irradiance = ZERO
  REAL(fp) :: Cosmic_Background_Radiance = ZERO
  REAL(fp) :: Planck_Surface = ZERO
  REAL(fp) :: cosmic_background = ZERO

  ! Input arrays
  REAL(fp), ALLOCATABLE :: w(:), T_OD(:)
  REAL(fp), ALLOCATABLE :: emissivity(:), direct_reflectivity(:)
  REAL(fp), ALLOCATABLE :: reflectivity(:,:)

  ! RT variables
  REAL(fp), ALLOCATABLE :: Planck_Atmosphere(:), total_opt(:)
  REAL(fp), ALLOCATABLE :: COS_Angle(:), COS_Weight(:), temporal_matrix(:,:)
  REAL(fp), ALLOCATABLE :: refl_down(:,:)
  REAL(fp), ALLOCATABLE :: s_Layer_Trans(:,:,:), s_Layer_Refl(:,:,:)
  REAL(fp), ALLOCATABLE :: s_Level_Refl_UP(:,:,:), s_Level_Rad_UP(:,:)
  REAL(fp), ALLOCATABLE :: s_Layer_Source_UP(:,:), s_Layer_Source_DOWN(:,:)

  ! Phase function variables
  REAL(fp), ALLOCATABLE :: Pff(:,:,:), Pbb(:,:,:)
  REAL(fp), ALLOCATABLE :: Pplus(:,:), Pminus(:,:), Pleg(:,:)
  REAL(fp), ALLOCATABLE :: Off(:,:,:), Obb(:,:,:)
  REAL(fp), ALLOCATABLE :: n_Factor(:,:), sum_fac(:,:)

  ! Adding-Doubling model
  REAL(fp), ALLOCATABLE :: Inv_Gamma(:,:,:), Inv_GammaT(:,:,:), Refl_Trans(:,:,:)

  ! AMOM-specific matrices
  REAL(fp), ALLOCATABLE :: Thermal_C(:,:), EigVa(:,:), Exp_x(:,:), EigValue(:,:)
  REAL(fp), ALLOCATABLE :: HH(:,:,:), PM(:,:,:), PP(:,:,:), PPM(:,:,:), PPP(:,:,:)
  REAL(fp), ALLOCATABLE :: i_PPM(:,:,:), i_PPP(:,:,:)
  REAL(fp), ALLOCATABLE :: EigVe(:,:,:), Gm(:,:,:), i_Gm(:,:,:), Gp(:,:,:)
  REAL(fp), ALLOCATABLE :: EigVeF(:,:,:), EigVeVa(:,:,:)
  REAL(fp), ALLOCATABLE :: A1(:,:,:), A2(:,:,:), A3(:,:,:), A4(:,:,:)
  REAL(fp), ALLOCATABLE :: A5(:,:,:), A6(:,:,:), Gm_A5(:,:,:), i_Gm_A5(:,:,:)
  
CONTAINS

  SUBROUTINE Initialize_CRTM_Shared()
    IMPLICIT NONE

    ALLOCATE(w(nL));                        w = ONE
    ALLOCATE(T_OD(nL));                     T_OD = TWO
    ALLOCATE(emissivity(nL));              emissivity = ZERO
    ALLOCATE(direct_reflectivity(nL));     direct_reflectivity = ZERO
    ALLOCATE(reflectivity(nA, nA));        reflectivity = ZERO

    ALLOCATE(Planck_Atmosphere(0:nL));     Planck_Atmosphere = ZERO
    ALLOCATE(total_opt(0:nL));             total_opt = ZERO
    ALLOCATE(COS_Angle(nA));               COS_Angle = ZERO
    ALLOCATE(COS_Weight(nA));              COS_Weight = ZERO
    ALLOCATE(temporal_matrix(nA,nA));      temporal_matrix = ZERO
    ALLOCATE(refl_down(nA,nL));            refl_down = ZERO

    ALLOCATE(s_Layer_Trans(nA,nA,nL));     s_Layer_Trans = ZERO
    ALLOCATE(s_Layer_Refl(nA,nA,nL));      s_Layer_Refl = ZERO
    ALLOCATE(s_Level_Refl_UP(nA,nA,0:nL)); s_Level_Refl_UP = ZERO
    ALLOCATE(s_Level_Rad_UP(nA,0:nL));     s_Level_Rad_UP = ZERO
    ALLOCATE(s_Layer_Source_UP(nA,nL));    s_Layer_Source_UP = ZERO
    ALLOCATE(s_Layer_Source_DOWN(nA,nL));  s_Layer_Source_DOWN = ZERO

    ALLOCATE(Pff(nA,nA+1,nL));             Pff = ZERO
    ALLOCATE(Pbb(nA,nA+1,nL));             Pbb = ZERO
    ALLOCATE(Pplus(0:nA,nA));              Pplus = ZERO
    ALLOCATE(Pminus(0:nA,nA));             Pminus = ZERO
    ALLOCATE(Pleg(0:nA,nA+1));             Pleg = ZERO
    ALLOCATE(Off(nA,nA+1,nL));             Off = ZERO
    ALLOCATE(Obb(nA,nA+1,nL));             Obb = ZERO
    ALLOCATE(n_Factor(nA,nL));             n_Factor = ZERO
    ALLOCATE(sum_fac(0:nA,nL));            sum_fac = ZERO

    ALLOCATE(Inv_Gamma(nA,nA,nL));         Inv_Gamma = ZERO
    ALLOCATE(Inv_GammaT(nA,nA,nL));        Inv_GammaT = ZERO
    ALLOCATE(Refl_Trans(nA,nA,nL));        Refl_Trans = ZERO

    ALLOCATE(Thermal_C(nA,nL));            Thermal_C = ZERO
    ALLOCATE(EigVa(nA,nL));                EigVa = ZERO
    ALLOCATE(Exp_x(nA,nL));                Exp_x = ZERO
    ALLOCATE(EigValue(nA,nL));             EigValue = ZERO
    ALLOCATE(HH(nA,nA,nL));                HH = ZERO
    ALLOCATE(PM(nA,nA,nL));                PM = ZERO
    ALLOCATE(PP(nA,nA,nL));                PP = ZERO
    ALLOCATE(PPM(nA,nA,nL));               PPM = ZERO
    ALLOCATE(PPP(nA,nA,nL));               PPP = ZERO
    ALLOCATE(i_PPM(nA,nA,nL));             i_PPM = ZERO
    ALLOCATE(i_PPP(nA,nA,nL));             i_PPP = ZERO
    ALLOCATE(EigVe(nA,nA,nL));             EigVe = ZERO
    ALLOCATE(Gm(nA,nA,nL));                Gm = ZERO
    ALLOCATE(i_Gm(nA,nA,nL));              i_Gm = ZERO
    ALLOCATE(Gp(nA,nA,nL));                Gp = ZERO
    ALLOCATE(EigVeF(nA,nA,nL));            EigVeF = ZERO
    ALLOCATE(EigVeVa(nA,nA,nL));           EigVeVa = ZERO
    ALLOCATE(A1(nA,nA,nL));                A1 = ZERO
    ALLOCATE(A2(nA,nA,nL));                A2 = ZERO
    ALLOCATE(A3(nA,nA,nL));                A3 = ZERO
    ALLOCATE(A4(nA,nA,nL));                A4 = ZERO
    ALLOCATE(A5(nA,nA,nL));                A5 = ZERO
    ALLOCATE(A6(nA,nA,nL));                A6 = ZERO
    ALLOCATE(Gm_A5(nA,nA,nL));             Gm_A5 = ZERO
    ALLOCATE(i_Gm_A5(nA,nA,nL));           i_Gm_A5 = ZERO

  END SUBROUTINE Initialize_CRTM_Shared

END MODULE CRTM_Shared

MODULE matrix_utils
  USE CRTM_Shared
  IMPLICIT NONE

CONTAINS

  FUNCTION matinv(a) RESULT(inv_mat)
    ! --------------------------------------------------------------------
    ! Compute the inversion of the matrix A
    ! Invert matrix by Gauss method
    ! --------------------------------------------------------------------

    REAL(fp), INTENT(IN), DIMENSION(:,:) :: a

    INTEGER :: n
    REAL(fp), DIMENSION(SIZE(a,1), SIZE(a,2)) :: b
    REAL(fp), DIMENSION(SIZE(a,1), SIZE(a,2)) :: inv_mat
    REAL(fp), DIMENSION(SIZE(a,1)) :: temp
    ! Local parameters
    CHARACTER(*), PARAMETER :: ROUTINE_NAME = 'matinv'
    ! - - - Local Variables - - -
    REAL(fp) :: c, d
    INTEGER :: i, j, k, m, imax(1), ipvt(SIZE(a,1))
    ! - - - - - - - - - - - - - -

    Error_Status = SUCCESS
    b = a
    n = SIZE(a,1)
    inv_mat = a
    ipvt = (/ (i, i = 1, n) /)

    ! Gauss-Jordan elimination for inversion
    DO k = 1, n
      ! Find the largest absolute value and its position in the column
      imax = MAXLOC(ABS(b(k:n,k)))
      m = k - 1 + imax(1)

      ! Singular matrix check
      IF (ABS(b(m,k)) <= 1.E-40_fp) THEN
        Error_Status = FAILURE
        CALL Display_Message(ROUTINE_NAME, 'Singular matrix', Error_Status)
        RETURN
      END IF

      ! Swap rows if needed
      IF (m /= k) THEN
        ipvt( (/ m,k /) ) = ipvt( (/ k,m /) )
        b((/ m,k /), :) = b((/ k,m /), :)
      END IF

      ! Scale pivot row
      d = 1.0_fp / b(k,k)
      temp = b(:,k)
      DO j = 1, n
        c = b(k,j) * d
        b(:,j) = b(:,j) - temp * c
        b(k,j) = c
      END DO
      b(:,k) = temp * (-d)
      b(k,k) = d
    END DO

    inv_mat(:, ipvt) = b

  END FUNCTION matinv

END MODULE matrix_utils

SUBROUTINE CRTM_ADA()
! ------------------------------------------------------------------------- !
! FUNCTION:                                                                 !
!   This subroutine calculates IR/MW radiance at the top of the atmosphere  !
!   including atmospheric scattering. The scheme will include solar part.   !
!   The ADA algorithm computes layer reflectance and transmittance as well  !
!   as source function by the subroutine CRTM_Doubling_layer, then uses     !
!   an adding method to integrate the layer and surface components.         !
!                                                                           ! 
!    Quanhua Liu    Quanhua.Liu@noaa.gov                                    !
! ------------------------------------------------------------------------- !

  USE CRTM_Shared
  USE matrix_utils
  IMPLICIT NONE

  ! -------------- internal variables --------------------------------- !
  !  Abbreviations:                                                     !
  !      s: scattering, rad: radiance, trans: transmission,             !
  !         refl: reflection, up: upward, down: downward                !
  ! --------------------------------------------------------------------!
  INTEGER :: i, j, k
  CHARACTER(*), PARAMETER :: ROUTINE_NAME = 'CRTM_ADA'
  CHARACTER(256) :: Message

  CALL Initialize_CRTM_Shared()
    
    total_opt(0) = ZERO
    DO k = 1, nL
      total_opt(k) = total_opt(k-1) + T_OD(k)
    END DO

    s_Layer_Trans = ZERO
    s_Layer_Refl = ZERO
    s_Level_Refl_UP = ZERO
    s_Level_Rad_UP = ZERO
    s_Layer_Source_UP = ZERO
    s_Layer_Source_DOWN = ZERO

    s_Level_Refl_UP(1:nA,1:nA,nL)=reflectivity(1:nA,1:nA)

    IF( mth_Azi == 0 ) THEN
      s_Level_Rad_UP(1:nA,nL ) = emissivity(1:nA)*Planck_Surface
    END IF

    IF( Solar_Flag_true ) THEN
      s_Level_Rad_UP(1:nA,nL ) = s_Level_Rad_UP(1:nA,nL )+direct_reflectivity(1:nA)* &
        COS_SUN*Solar_irradiance/PI*exp(-total_opt(nL)/COS_SUN)       
    END IF

    ! UPWARD ADDING LOOP STARTS FROM BOTTOM LAYER TO ATMOSPHERIC TOP LAYER.
    DO 10 k = nL, 1, -1

    ! Compute tranmission and reflection matrices for a layer
    IF(w(k) > SCATTERING_ALBEDO_THRESHOLD) THEN 

    !  ----------------------------------------------------------- !
    !    CALL  multiple-stream algorithm for computing layer       !
    !    transmission, reflection, and source functions.           !
    !  ----------------------------------------------------------- !

       CALL CRTM_AMOM_layer(k)! Input
       
    ! Internal variable  
    !  ----------------------------------------------------------- !
    !    Adding method to add the layer to the present level       !
    !    to compute upward radiances and reflection matrix         !
    !    at new level.                                             !
    !  ----------------------------------------------------------- !

    
    temporal_matrix = -matmul(s_Level_Refl_UP(1:nA,1:nA,k),  &
                       s_Layer_Refl(1:nA,1:nA,k))
    DO i = 1, nA 
      temporal_matrix(i,i) = ONE + temporal_matrix(i,i)
    END DO

    Inv_Gamma(1:nA,1:nA,k) = matinv(temporal_matrix)
    IF( Error_Status /= SUCCESS  ) THEN
      WRITE( Message,'("Error in matrix inversion matinv(temporal_matrix, Error_Status) ")' ) 
      RETURN                                                                                    
    END IF
         
    Inv_GammaT(1:nA,1:nA,k) =   &
     matmul(s_Layer_Trans(1:nA,1:nA,k), Inv_Gamma(1:nA,1:nA,k))
    refl_down(1:nA,k) = matmul(s_Level_Refl_UP(1:nA,1:nA,k),  &
                                  s_Layer_Source_DOWN(1:nA,k))

    s_Level_Rad_UP(1:nA,k-1 )=s_Layer_Source_UP(1:nA,k)+ &
    matmul(Inv_GammaT(1:nA,1:nA,k),refl_down(1:nA,k) &
          +s_Level_Rad_UP(1:nA,k ))
    Refl_Trans(1:nA,1:nA,k) = matmul(s_Level_Refl_UP(1:nA,1:nA,k), &
          s_Layer_Trans(1:nA,1:nA,k))
    s_Level_Refl_UP(1:nA,1:nA,k-1)=s_Layer_Refl(1:nA,1:nA,k) + &
    matmul(Inv_GammaT(1:nA,1:nA,k),Refl_Trans(1:nA,1:nA,k)) 

    ELSE
      DO i = 1, nA 
        s_Layer_Trans(i,i,k) = exp(-T_OD(k)/COS_Angle(i))
        s_Layer_Source_UP(i,k) = Planck_Atmosphere(k) * (ONE - s_Layer_Trans(i,i,k) )
        s_Layer_Source_DOWN(i,k) = s_Layer_Source_UP(i,k)

      END DO

      ! Adding method
      DO i = 1, nA 
        s_Level_Rad_UP(i,k-1 )=s_Layer_Source_UP(i,k)+ &
        s_Layer_Trans(i,i,k)*(sum(s_Level_Refl_UP(i,1:nA,k)*s_Layer_Source_DOWN(1:nA,k))  &
         +s_Level_Rad_UP(i,k ))
      ENDDO
        DO i = 1, nA 
          DO j = 1, nA 
            s_Level_Refl_UP(i,j,k-1)=s_Layer_Trans(i,i,k)*s_Level_Refl_UP(i,j,k)*s_Layer_Trans(j,j,k)
          ENDDO
        ENDDO
    ENDIF
    
    10     CONTINUE

    !  Adding reflected cosmic background radiation
    IF( mth_Azi == 0 ) THEN
       DO i = 1, nA 
         s_Level_Rad_UP(i,0)=s_Level_Rad_UP(i,0)+sum(s_Level_Refl_UP(i,1:nA,0))*cosmic_background
       ENDDO
    END IF

    RETURN
    
  END SUBROUTINE CRTM_ADA 
      
  SUBROUTINE CRTM_AMOM_layer(KL) !Input, KL-th layer 
  
! ---------------------------------------------------------------------------------------
!   FUNCTION
!    Compute layer transmission, reflection matrices and source function 
!    at the top and bottom of the layer.
!
!   Method and References
!    The transmittance and reflectance matrices is further derived from 
!    matrix operator method. The matrix operator method is referred to the paper by
!
!    Weng, F., and Q. Liu, 2003: Satellite Data Assimilation in Numerical Weather Prediction
!    Model: Part 1: Forward Radiative Transfer and Jacobian Modeling in Cloudy Atmospheres,
!    J. Atmos. Sci., 60, 2633-2646.
!
!   see also ADA method.
!   Quanhua Liu
!   Quanhua.Liu@noaa.gov
    ! ----------------------------------------------------------------------------------------

    USE CRTM_Shared
    USE matrix_utils
    
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: KL

   ! internal variables
   REAL(fp), DIMENSION(nA,nA) :: trans, refl, tempo
   REAL(fp) :: s, c, xx
   INTEGER :: i,j,N2,N2_1
   REAL(fp) :: EXPfactor,Sfactor,s_transmittance,Solar(2*nA),V0(2*nA,2*nA),Solar1(2*nA)
   REAL(fp) :: V1(2*nA,2*nA),Sfac2,source_up(nA),source_down(nA)    
   CHARACTER(256) :: Message

   ! for small layer optical depth, single scattering is applied.  
   IF( T_OD(KL) < DELTA_OPTICAL_DEPTH ) THEN
     s = T_OD(KL) * w(KL)
     DO i = 1, nA
       Thermal_C(i,KL) = ZERO
       c = s/COS_Angle(i)
       DO j = 1, nA
         s_Layer_Refl(i,j,KL) = c * Pbb(i,j,KL) * COS_Weight(j)
         s_Layer_Trans(i,j,KL) = c * Pff(i,j,KL) * COS_Weight(j)
         IF( i == j ) THEN
           s_Layer_Trans(i,i,KL) = s_Layer_Trans(i,i,KL) + &
             ONE - T_OD(KL)/COS_Angle(i)
         END IF
         IF( mth_Azi == 0 ) THEN
           Thermal_C(i,KL) = Thermal_C(i,KL) + &
           ( s_Layer_Refl(i,j,KL) + s_Layer_Trans(i,j,KL) )
         END IF
       ENDDO

       IF( mth_Azi == 0 ) THEN
         s_Layer_Source_UP(i,KL) = ( ONE - Thermal_C(i,KL) ) * Planck_Atmosphere(KL)
         s_Layer_Source_DOWN(i,KL) = s_Layer_Source_UP(i,KL)
       END IF
     ENDDO

     RETURN

   END IF
   !
   ! for numerical stability, 
   IF( w(KL) < max_albedo ) THEN
     s = w(KL)
   ELSE
     s = max_albedo
   END IF
   !
   ! building phase matrices
   DO i = 1, nA
     c = s/COS_Angle(i)
     DO j = 1, nA
       PM(i,j,KL) = c * Pbb(i,j,KL) * COS_Weight(j)
       PP(i,j,KL) = c * Pff(i,j,KL) * COS_Weight(j)
     ENDDO
       PP(i,i,KL) = PP(i,i,KL) - ONE/COS_Angle(i)
   ENDDO
   PPM(1:nA,1:nA,KL) = PP(1:nA,1:nA,KL) - PM(1:nA,1:nA,KL)
   i_PPM(1:nA,1:nA,KL) = matinv( PPM(1:nA,1:nA,KL))
   IF( Error_Status /= SUCCESS  ) THEN
     WRITE( Message,'("Error in matrix inversion matinv( PPM(1:nA,1:nA,KL), Error_Status ) ")' ) 

     RETURN                                                                                    
   END IF

   PPP(1:nA,1:nA,KL) = PP(1:nA,1:nA,KL) + PM(1:nA,1:nA,KL)
   HH(1:nA,1:nA,KL) = matmul( PPM(1:nA,1:nA,KL), PPP(1:nA,1:nA,KL) )   
   !
   ! save phase element HH, call ASYMTX for calculating eigenvalue and vectors.
   tempo = HH(1:nA,1:nA,KL)
   CALL ASYMTX(tempo,nA,nA,nA,EigVe(1:nA,1:nA,KL),EigVa(1:nA,KL),Error_Status)
   DO i = 1, nA
     IF( EigVa(i,KL) > ZERO ) THEN         
       EigValue(i,KL) = sqrt( EigVa(i,KL) )
     ELSE
       EigValue(i,KL) = ZERO
     END IF
   END DO

   DO i = 1, nA
     DO j = 1, nA
       EigVeVa(i,j,KL) = EigVe(i,j,KL) * EigValue(j,KL)
     END DO
   END DO       
   EigVeF(1:nA,1:nA,KL) = matmul( i_PPM(1:nA,1:nA,KL), EigVeVa(1:nA,1:nA,KL) )

   ! compute layer reflection, transmission and source function
   Gp(1:nA,1:nA,KL) = ( EigVe(1:nA,1:nA,KL) + EigVeF(1:nA,1:nA,KL) )/2.0_fp
   Gm(1:nA,1:nA,KL) = ( EigVe(1:nA,1:nA,KL) - EigVeF(1:nA,1:nA,KL) )/2.0_fp
   i_Gm(1:nA,1:nA,KL) = matinv( Gm(1:nA,1:nA,KL))

   IF( Error_Status /= SUCCESS  ) THEN
     WRITE( Message,'("Error in matrix inversion matinv( Gm(1:nA,1:nA,KL), Error_Status) ")' ) 
     RETURN                                                                                    
   END IF             

   DO i = 1, nA
     xx = EigValue(i,KL)*T_OD(KL)
     Exp_x(i,KL) = exp(-xx)
   END DO

   DO i = 1, nA
     DO j = 1, nA
       A1(i,j,KL) = Gp(i,j,KL) * Exp_x(j,KL)
       A4(i,j,KL) = Gm(i,j,KL) * Exp_x(j,KL)
     END DO
   END DO

   A2(1:nA,1:nA,KL) = matmul( i_Gm(1:nA,1:nA,KL), A1(1:nA,1:nA,KL) )
   A3(1:nA,1:nA,KL) = matmul( Gp(1:nA,1:nA,KL), A2(1:nA,1:nA,KL) )
   A5(1:nA,1:nA,KL) = matmul( A1(1:nA,1:nA,KL), A2(1:nA,1:nA,KL) )
   A6(1:nA,1:nA,KL) = matmul( A4(1:nA,1:nA,KL), A2(1:nA,1:nA,KL) )
   Gm_A5(1:nA,1:nA,KL) = Gm(1:nA,1:nA,KL) - A5(1:nA,1:nA,KL)     
   i_Gm_A5(1:nA,1:nA,KL) = matinv(Gm_A5(1:nA,1:nA,KL))
   IF( Error_Status /= SUCCESS  ) THEN
     WRITE( Message,'("Error in matrix inversion matinv(Gm_A5(1:nA,1:nA,KL), Error_Status) ")' ) 
     RETURN                                                                                    
   END IF
   trans = matmul( A4(1:nA,1:nA,KL) - A3(1:nA,1:nA,KL), i_Gm_A5(1:nA,1:nA,KL) ) 
   refl = matmul( Gp(1:nA,1:nA,KL) - A6(1:nA,1:nA,KL), i_Gm_A5(1:nA,1:nA,KL) )

   ! post processing  
   s_Layer_Trans(1:nA,1:nA,KL) = trans(:,:)
   s_Layer_Refl(1:nA,1:nA,KL) = refl(:,:)
   s_Layer_Source_UP(:,KL) = ZERO
   IF( mth_Azi == 0 ) THEN
     DO i = 1, nA
       Thermal_C(i,KL) = ZERO
       DO j = 1, nS
         Thermal_C(i,KL) = Thermal_C(i,KL) + (trans(i,j) + refl(i,j) )
       END DO
       IF ( i == nA .AND. nA == (nS+1) ) THEN
         Thermal_C(i,KL) = Thermal_C(i,KL) + trans(nA,nA)
       END IF
       s_Layer_Source_UP(i,KL) = ( ONE - Thermal_C(i,KL) ) * Planck_Atmosphere(KL)
       s_Layer_Source_DOWN(i,KL) = s_Layer_Source_UP(i,KL)
     END DO
   END IF

   !  compute visible part for visible channels during daytime
   IF( Solar_Flag_true ) THEN
     N2 = 2 * nA
     N2_1 = N2 - 1
     source_up = ZERO
     source_down = ZERO
     !
     ! Solar source  
     Sfactor = w(KL)*Solar_irradiance/PI
     IF( mth_Azi == 0 ) Sfactor = Sfactor/TWO
       EXPfactor = exp(-T_OD(KL)/COS_SUN)
       s_transmittance = exp(-total_opt(KL-1)/COS_SUN)

       DO i = 1, nA     
         Solar(i) = -Pbb(i,nA+1,KL)*Sfactor
         Solar(i+nA) = -Pff(i,nA+1,KL)*Sfactor

         DO j = 1, nA
           V0(i,j) = w(KL) * Pff(i,j,KL) * COS_Weight(j)
           V0(i+nA,j) = w(KL) * Pbb(i,j,KL) * COS_Weight(j)
           V0(i,j+nA) = V0(i+nA,j)
           V0(nA+i,j+nA) = V0(i,j)
         ENDDO
         V0(i,i) = V0(i,i) - ONE - COS_Angle(i)/COS_SUN
         V0(i+nA,i+nA) = V0(i+nA,i+nA) - ONE + COS_Angle(i)/COS_SUN
       ENDDO

       V1(1:N2_1,1:N2_1) = matinv(V0(1:N2_1,1:N2_1))
       IF( Error_Status /= SUCCESS  ) THEN
         WRITE( Message,'("Error in matrix inversion matinv(V0(1:N2_1,1:N2_1), Error_Status) ")' ) 
         RETURN                                                                                    
       END IF         

       Solar1(1:N2_1) = matmul( V1(1:N2_1,1:N2_1), Solar(1:N2_1) )
       Solar1(N2) = ZERO
       Sfac2 = Solar(N2) - sum( V0(N2,1:N2_1)*Solar1(1:N2_1) )


       DO i = 1, nA
         source_up(i) = Solar1(i)
         source_down(i) = EXPfactor*Solar1(i+nA)
         DO j = 1, nA
          source_up(i) =source_up(i)-refl(i,j)*Solar1(j+nA)-trans(i,j)*EXPfactor*Solar1(j)
          source_down(i) =source_down(i) -trans(i,j)*Solar1(j+nA) -refl(i,j)*EXPfactor*Solar1(j)
         END DO
       END DO
       ! specific treatment for downeward source function
       IF( abs( V0(N2,N2) ) > 0.0001_fp ) THEN
         source_down(nA) =source_down(nA) +(EXPfactor-trans(nA,nA))*Sfac2/V0(N2,N2)
       ELSE
         source_down(nA) =source_down(nA) -EXPfactor*Sfac2*T_OD(KL)/COS_Angle(nA)
       END IF

       source_up(1:nA) = source_up(1:nA)*s_transmittance
       source_down(1:nA) = source_down(1:nA)*s_transmittance

       s_Layer_Source_UP(1:nA,KL) = s_Layer_Source_UP(1:nA,KL)+source_up(1:nA)
       s_Layer_Source_DOWN(1:nA,KL) = s_Layer_Source_DOWN(1:nA,KL)+source_down(1:nA)
   END IF

   RETURN

  END SUBROUTINE CRTM_AMOM_layer
