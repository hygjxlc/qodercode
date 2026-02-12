package com.example.expert.repository;

import com.example.expert.entity.ContactRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ContactRecordRepository extends JpaRepository<ContactRecord, Long> {
    
    List<ContactRecord> findByExpertIdOrderByContactTimeDesc(Long expertId);
}
