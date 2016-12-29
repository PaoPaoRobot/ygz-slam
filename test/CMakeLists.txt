add_executable( test_tum_tracking test_tum_tracking.cpp )
target_link_libraries( test_tum_tracking 
    ygz-slam ${THIRD_PARTY_LIBS}
)

add_executable( test_tum_vo test_tum_vo.cpp )
target_link_libraries( test_tum_vo 
    ygz-slam ${THIRD_PARTY_LIBS}
)

add_executable( test_orb_match test_orb_match.cpp )
target_link_libraries( test_orb_match 
    ygz-slam ${THIRD_PARTY_LIBS}
)