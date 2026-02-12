package com.example.expert.repository;

import com.example.expert.entity.Expert;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface ExpertRepository extends JpaRepository<Expert, Long> {
    
    Optional<Expert> findByEmailAndIsDeletedFalse(String email);
    
    Optional<Expert> findByIdAndIsDeletedFalse(Long id);
    
    boolean existsByEmailAndIsDeletedFalse(String email);
    
    Page<Expert> findByIsDeletedFalse(Pageable pageable);
    
    long countByIsDeletedFalse();
    
    @Query("SELECT e FROM Expert e WHERE e.isDeleted = false AND " +
           "(LOWER(e.name) LIKE LOWER(CONCAT('%', :keyword, '%')) OR " +
           "LOWER(e.professionalField) LIKE LOWER(CONCAT('%', :keyword, '%')))");
    List<Expert> searchByKeyword(@Param("keyword") String keyword);
    
    @Query("SELECT e FROM Expert e WHERE e.isDeleted = false AND " +
           "(:field IS NULL OR e.professionalField = :field) AND " +
           "(:title IS NULL OR e.title = :title) AND " +
           "(:status IS NULL OR e.contactStatus = :status)")
    List<Expert> filterExperts(@Param("field") String field, 
                               @Param("title") String title, 
                               @Param("status") String status);
}
