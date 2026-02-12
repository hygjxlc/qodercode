package com.example.expert.repository;

import com.example.expert.entity.ConferenceRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface ConferenceRecordRepository extends JpaRepository<ConferenceRecord, Long> {
    
    List<ConferenceRecord> findByExpertIdOrderByConferenceDateDesc(Long expertId);
    
    Optional<ConferenceRecord> findByExpertIdAndConferenceNameAndConferenceDate(
            Long expertId, String conferenceName, LocalDate conferenceDate);
}
