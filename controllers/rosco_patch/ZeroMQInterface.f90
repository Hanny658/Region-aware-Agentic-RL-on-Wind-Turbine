module ZeroMQInterface
   ! PATCHED (region-aware RL project): measurement vector extended 17 -> 22.
   ! Extra channels (18..22) expose the ROSCO-native pitch command, the current
   ! lower pitch limit (fine pitch / peak-shaving), the measured blade pitch,
   ! the estimated wind speed and the final applied pitch command, so that the
   ! Python side can (a) build the RL state, (b) clamp the residual against
   ! PC_MinPit, and (c) label the operating region. Setpoints are unchanged.
   USE, INTRINSIC :: ISO_C_BINDING, only: C_CHAR, C_DOUBLE, C_NULL_CHAR
   IMPLICIT NONE

CONTAINS
    SUBROUTINE UpdateZeroMQ(LocalVar, CntrPar, ErrVar)
        USE ROSCO_Types, ONLY : LocalVariables, ControlParameters, ErrorVariables
        IMPLICIT NONE
        TYPE(LocalVariables),    INTENT(INOUT) :: LocalVar
        TYPE(ControlParameters), INTENT(INOUT) :: CntrPar
        TYPE(ErrorVariables),    INTENT(INOUT) :: ErrVar

        character(256) :: zmq_address
        real(C_DOUBLE) :: setpoints(8)
        real(C_DOUBLE) :: turbine_measurements(22)
        CHARACTER(*), PARAMETER                 :: RoutineName = 'UpdateZeroMQ'

#ifdef ZMQ_CLIENT
        interface
            subroutine zmq_client(zmq_address, measurements, setpoints) bind(C, name='zmq_client')
                import :: C_CHAR, C_DOUBLE
                implicit none
                character(C_CHAR), intent(out) :: zmq_address(*)
                real(C_DOUBLE) :: measurements(22)
                real(C_DOUBLE) :: setpoints(8)
            end subroutine zmq_client
        end interface
#endif

        IF ( MOD(LocalVar%n_DT, CntrPar%n_DT_ZMQ) == 0 .OR. LocalVar%iStatus == -1 ) THEN
            ! --- stock ROSCO measurements (1..17) ---
            turbine_measurements(1) = LocalVar%ZMQ_ID
            turbine_measurements(2) = LocalVar%iStatus
            turbine_measurements(3) = LocalVar%Time
            turbine_measurements(4) = LocalVar%VS_MechGenPwr
            turbine_measurements(5) = LocalVar%VS_GenPwr
            turbine_measurements(6) = LocalVar%GenSpeed
            turbine_measurements(7) = LocalVar%RotSpeed
            turbine_measurements(8) = LocalVar%GenTqMeas
            turbine_measurements(9) = LocalVar%NacHeading
            turbine_measurements(10) = LocalVar%NacVane
            turbine_measurements(11) = LocalVar%HorWindV
            turbine_measurements(12) = LocalVar%rootMOOP(1)
            turbine_measurements(13) = LocalVar%rootMOOP(2)
            turbine_measurements(14) = LocalVar%rootMOOP(3)
            turbine_measurements(15) = LocalVar%FA_Acc_TT
            turbine_measurements(16) = LocalVar%NacIMU_FA_RAcc
            turbine_measurements(17) = LocalVar%Azimuth
            ! --- project additions (18..22) ---
            turbine_measurements(18) = LocalVar%BlPitch(1)       ! measured blade-1 pitch [rad]
            turbine_measurements(19) = LocalVar%PC_PitComT       ! ROSCO collective pitch cmd, before ZMQ offset [rad]
            turbine_measurements(20) = LocalVar%PC_MinPit        ! current lower pitch limit (fine pitch / peak shaving) [rad]
            turbine_measurements(21) = LocalVar%WE_Vw            ! ROSCO wind speed estimate [m/s]
            turbine_measurements(22) = LocalVar%PitCom(1)        ! final applied blade-1 pitch cmd (incl. offset, rate-limited) [rad]

            write (zmq_address, '(A,A)') TRIM(CntrPar%ZMQ_CommAddress), C_NULL_CHAR
#ifdef ZMQ_CLIENT
            call zmq_client(zmq_address, turbine_measurements, setpoints)
#else
            ErrVar%aviFAIL = -1
            IF (CntrPar%ZMQ_Mode > 0) THEN
                ErrVar%ErrMsg = ' >> The ZeroMQ client has not been properly installed, ' &
                                //'please install it to use ZMQ_Mode > 0.'
                ErrVar%ErrMsg = RoutineName//':'//TRIM(ErrVar%ErrMsg)
            ENDIF
#endif

            LocalVar%ZMQ_TorqueOffset = setpoints(1)
            LocalVar%ZMQ_YawOffset = setpoints(2)
            LocalVar%ZMQ_PitOffset(1) = setpoints(3)
            LocalVar%ZMQ_PitOffset(2) = setpoints(4)
            LocalVar%ZMQ_PitOffset(3) = setpoints(5)
            LocalVar%ZMQ_R_Speed = setpoints(6)
            LocalVar%ZMQ_R_Torque = setpoints(7)
            LocalVar%ZMQ_R_Pitch = setpoints(8)
        ENDIF

    END SUBROUTINE UpdateZeroMQ
end module ZeroMQInterface
